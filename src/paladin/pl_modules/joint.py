import logging
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import hydra
import lightning.pytorch as pl
import numpy as np
import torch
import torchmetrics
from scipy.stats import beta
from torch.optim import Optimizer

from nn_core.model_logging import NNLogger

from paladin.data.joint_datamodule import MetaData
from paladin.pl_modules.callbacks import ExportCallback

pylogger = logging.getLogger(__name__)


class AbstractJointLightningModule(pl.LightningModule):
    logger: NNLogger

    def __init__(self, metadata: Optional[MetaData] = None, *args, **kwargs) -> None:
        super().__init__()

        # Populate self.hparams with args and kwargs automagically!
        # We want to skip metadata since it is saved separately by the NNCheckpointIO object.
        # Be careful when modifying this instruction. If in doubt, don't do it :]
        self.save_hyperparameters(logger=False, ignore=("metadata",))
        self.learning_rate = self.hparams.optimizer.lr
        self.metadata = metadata

        self.metrics = self.initialize_metrics()

    def initialize_metrics(self) -> torchmetrics.MetricCollection:
        metrics = {}
        for split in ["train", "val", "test"]:
            metrics[f"{split}_loss"] = torchmetrics.aggregation.MeanMetric()
            if self.metadata.n_clf_tasks > 1:
                metrics[f"{split}_auroc"] = torchmetrics.classification.MultilabelAUROC(
                    num_labels=self.metadata.n_clf_tasks
                )
            elif self.metadata.n_clf_tasks == 1:
                metrics[f"{split}_auroc"] = torchmetrics.classification.AUROC(task="binary")

            if self.metadata.n_reg_tasks > 0:
                metrics[f"{split}_mse"] = torchmetrics.MeanSquaredError(num_outputs=self.metadata.n_reg_tasks)
                metrics[f"{split}_pearson"] = torchmetrics.PearsonCorrCoef(num_outputs=self.metadata.n_reg_tasks)

        return torchmetrics.MetricCollection(metrics)

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Method for the forward pass.

        Args:
            batch: input batch

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        # example
        return self.model(batch)

    def calculate_loss(self, logits, gt_y, split):
        if self.metadata.n_clf_tasks > 0:
            # print(f"logits: {logits.shape}, gt_y: {gt_y.shape}")
            logits_clf = logits[:, : self.metadata.n_clf_tasks]
            gt_y_clf = gt_y[:, : self.metadata.n_clf_tasks]
            # print(f"logits_clf: {logits_clf.shape}, gt_y_clf: {gt_y_clf.shape}")
            clf_loss_value = self.clf_loss(logits_clf, gt_y_clf)

        if self.metadata.n_reg_tasks > 0:
            logits_reg = logits[:, self.metadata.n_clf_tasks :]
            gt_y_reg = gt_y[:, self.metadata.n_clf_tasks :]
            reg_loss_value = self.reg_loss(logits_reg, gt_y_reg)

        # combine clf and reg losses
        if self.metadata.n_clf_tasks > 0 and self.metadata.n_reg_tasks > 0:
            loss_value = clf_loss_value + self.balance * reg_loss_value
        elif self.metadata.n_clf_tasks > 0:
            loss_value = clf_loss_value
        elif self.metadata.n_reg_tasks > 0:
            loss_value = reg_loss_value
        else:
            raise ValueError(
                f"Metadata has {self.metadata.n_clf_tasks} classification tasks and {self.metadata.n_reg_tasks} regression tasks"
            )
        return loss_value

    def log_metrics(self, logits, gt_y, loss_value, batch_size, split):
        self.metrics[f"{split}_loss"].update(loss_value)

        if self.metadata.n_clf_tasks > 0:
            self.metrics[f"{split}_auroc"].update(
                logits[:, : self.metadata.n_clf_tasks], gt_y[:, : self.metadata.n_clf_tasks].int()
            )
        if self.metadata.n_reg_tasks > 0:
            self.metrics[f"{split}_mse"].update(
                logits[:, self.metadata.n_clf_tasks :], gt_y[:, self.metadata.n_clf_tasks :]
            )
            self.metrics[f"{split}_pearson"].update(
                logits[:, self.metadata.n_clf_tasks :], gt_y[:, self.metadata.n_clf_tasks :]
            )

        log_dict = {f"loss/{split}": loss_value}
        if self.metadata.n_clf_tasks > 0:
            log_dict[f"auroc/{split}"] = self.metrics[f"{split}_auroc"].compute()
        if self.metadata.n_reg_tasks > 0:
            log_dict[f"mse/{split}"] = self.metrics[f"{split}_mse"].compute().mean()
            log_dict[f"pearson/{split}"] = self.metrics[f"{split}_pearson"].compute().mean()
        self.log_dict(log_dict, on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)

    def _step(self, batch: Dict[str, torch.Tensor], split: str) -> Mapping[str, torch.Tensor]:
        gt_y = batch[self.hparams.y_key]

        results = self(batch)
        logits = results["logits"]
        loss_value = self.calculate_loss(logits, gt_y, split)

        if any(isinstance(cb, ExportCallback) for cb in self.trainer.callbacks):
            ExportCallback.update_forward_outputs(self, batch, results, split)

        if split != "test":
            self.log_metrics(logits, gt_y, loss_value, len(batch["image_ids"]), split)

        return {"logits": logits.detach(), "loss": loss_value}

    def training_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        return self._step(batch=batch, split="train")

    def _collect_base_outputs(self, output_dict, batch, logits):
        """Collect common metadata fields shared by all subclasses."""
        output_dict["histology"].extend(batch["oncotree_code"])
        output_dict["sample_id"].extend(batch["sample_id"])
        output_dict["split"].extend(batch["split"])

    def validation_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        results = self._step(batch=batch, split="val")
        if self.calibrator is not None:
            results["prob"] = self.calibrator.predict_proba(results["logits"].cpu().numpy().reshape(-1, 1))[
                :, 1
            ].tolist()
        else:
            results["prob"] = [-1] * len(results["logits"])

        self.validation_outputs["logits"].extend([x.cpu().numpy().reshape(-1) for x in results["logits"]])
        self.validation_outputs["prob"].extend(results["prob"])
        self.validation_outputs["target"].extend(batch[self.hparams.y_key].cpu().numpy().reshape(-1).tolist())
        self._collect_base_outputs(self.validation_outputs, batch, results["logits"])
        return results

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Mapping[str, Any]:
        result = self._step(batch=batch, split="test")
        if self.calibrator is not None:
            # TODO: update to handle multiple tasks
            result["prob"] = self.calibrator.predict_proba(result["logits"].cpu().numpy().reshape(-1, 1))[:, 1].tolist()
        else:
            # If multiple tasks, extend with -1 for all tasks per sample
            total_tasks = self.metadata.n_clf_tasks + self.metadata.n_reg_tasks
            result["prob"] = [-1] * (result["logits"].shape[0] * total_tasks)

        # Convert logits and probs to flat lists
        logits_np = result["logits"].cpu().numpy()  # Shape: [batch_size, total_tasks]
        probs_np = np.array(result["prob"]).reshape(-1, self.metadata.n_clf_tasks + self.metadata.n_reg_tasks)

        # Flatten logits and probs
        self.test_outputs["logits"].extend(logits_np.flatten().tolist())
        self.test_outputs["prob"].extend(probs_np.flatten().tolist())
        self.test_outputs["target"].extend(batch[self.hparams.y_key].cpu().numpy().flatten().tolist())

        # Repeat 'histology', 'sample_id', and 'split' for each task
        total_tasks = self.metadata.n_clf_tasks + self.metadata.n_reg_tasks
        for field in ["histology", "sample_id", "split"]:
            if field == "histology":
                values = batch["oncotree_code"]
            else:
                values = batch[field]
            if isinstance(values, torch.Tensor):
                values = values.cpu().numpy()
            # Repeat each value 'total_tasks' times
            extended_values = np.repeat(values, total_tasks).tolist()
            self.test_outputs[field].extend(extended_values)

        return {"logits": logits_np, "loss": result["loss"]}

    def on_test_epoch_start(self) -> None:
        self.test_outputs = {"logits": [], "prob": [], "target": [], "histology": [], "sample_id": [], "split": []}

    def on_validation_epoch_start(self) -> None:
        self.validation_outputs = {
            "logits": [],
            "prob": [],
            "target": [],
            "histology": [],
            "sample_id": [],
            "split": [],
        }

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        opt = hydra.utils.instantiate(
            self.hparams.optimizer, params=self.parameters(), _convert_="partial", lr=self.learning_rate
        )  # to be able to pass the learning rate after tuning
        if "lr_scheduler" not in self.hparams:
            return [opt]
        scheduler = hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer=opt)
        return [opt], [scheduler]


class JointLightningModule(AbstractJointLightningModule):
    logger: NNLogger

    def __init__(self, model, metadata, *args, **kwargs):
        super().__init__(metadata, *args, **kwargs)
        self.calibrator = None

        pos_weights = self.metadata.target_dict["pos_weights"]
        pos_weights = torch.tensor(pos_weights).reshape(1, -1).to(self.device)

        self.clf_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction="mean")
        self.reg_loss = torch.nn.MSELoss(reduction="mean")
        if self.metadata.n_clf_tasks == 0:
            self.balance = 1.0
        else:
            self.balance = self.metadata.n_reg_tasks / self.metadata.n_clf_tasks

        print(f"n_clf_tasks: {self.metadata.n_clf_tasks}")
        print(f"n_reg_tasks: {self.metadata.n_reg_tasks}")

        self.model = hydra.utils.instantiate(
            model,
            num_targets=self.metadata.n_reg_tasks + self.metadata.n_clf_tasks,
            _recursive_=False,
        )


class JointBetaBinomialLightningModule(AbstractJointLightningModule):
    logger: NNLogger

    def __init__(self, model, metadata, *args, **kwargs):
        super().__init__(metadata, *args, **kwargs)

        assert self.metadata.n_clf_tasks == 1, "Only one classification task is supported for beta-binomial loss"
        assert self.metadata.n_reg_tasks == 0, "No regression tasks are supported for beta-binomial loss"

        pos_weights = self.metadata.target_dict["pos_weights"]
        self.pos_weights = torch.tensor(pos_weights).reshape(1, -1).to(self.device)

        self.clf_loss = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weights, reduction="mean")
        self.reg_loss = torch.nn.MSELoss(reduction="mean")
        self.balance = 0.0  # No regression tasks

        print(f"n_clf_tasks: {self.metadata.n_clf_tasks}")
        print(f"n_reg_tasks: {self.metadata.n_reg_tasks}")

        self.model = hydra.utils.instantiate(
            model,
            num_targets=2 * (self.metadata.n_reg_tasks + self.metadata.n_clf_tasks),
            _recursive_=False,
        )

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Method for the forward pass for beta-binomial loss.

        Args:
            batch: input batch

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        # example
        output = self.model(batch)
        logits = output["logits"]

        # Apply softplus to ensure positive values for beta-binomial parameters
        logits = torch.nn.functional.softplus(logits) + 1.0  # to enforce concavity
        output["logits"] = logits
        return output

    def calculate_loss(self, logits, gt_y, split, pos_weights=None):
        # logits is a tensor of shape (batch_size, 2 * (n_clf_tasks + n_reg_tasks))
        # gt_y is a tensor of shape (batch_size, n_clf_tasks + n_reg_tasks)
        # logits alternate between alpha and beta parameters for the beta distribution
        # odds are the a, evens are the b
        a = logits[:, ::2]
        b = logits[:, 1::2]

        # calculate the beta-binomial loss
        loss = (
            -torch.lgamma(gt_y + a)  # noqa
            - torch.lgamma(1 - gt_y + b)  # noqa
            + torch.lgamma(1 + a + b)  # noqa
            - torch.lgamma(a + b)  # noqa
            + torch.lgamma(a)  # noqa
            + torch.lgamma(b)  # noqa
        )
        # pos_weights applied here
        if pos_weights is not None:
            assert pos_weights.ndim == 2
            assert pos_weights.shape[0] == 1
            assert pos_weights.shape[1] == 1
            loss[gt_y == 1] *= pos_weights.item()
        return loss.mean()

    def logits_to_point_estimates(self, logits):
        # logits is a tensor of shape (batch_size, 2 * (n_clf_tasks + n_reg_tasks))
        # need to convert it to a tensor of shape (batch_size, n_clf_tasks + n_reg_tasks)
        return logits[:, ::2] / (logits[:, ::2] + logits[:, 1::2])

    def log_metrics(self, logits, gt_y, loss_value, batch_size, split):
        # logits is a tensor of shape (batch_size, 2 * (n_clf_tasks + n_reg_tasks))
        # need to convert it to a tensor of shape (batch_size, n_clf_tasks + n_reg_tasks)
        assert self.metadata.n_clf_tasks == 1, "Only one classification task is supported for beta-binomial loss"
        assert self.metadata.n_reg_tasks == 0, "No regression tasks are supported for beta-binomial loss"
        point_estimates = self.logits_to_point_estimates(logits)
        super().log_metrics(point_estimates, gt_y, loss_value, batch_size, split)

    def validation_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        results = self._step(batch=batch, split="val")
        return results

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Mapping[str, Any]:
        results = self._step(batch=batch, split="test")
        point_estimates = self.logits_to_point_estimates(results["logits"])
        lower_bounds, upper_bounds = self.logits_to_bounds(results["logits"])
        self.test_outputs["logits"].extend([x.cpu().numpy().reshape(-1) for x in point_estimates])
        self.test_outputs["target"].extend(batch[self.hparams.y_key].cpu().numpy().reshape(-1).tolist())
        self._collect_base_outputs(self.test_outputs, batch, results["logits"])
        self.test_outputs["lower_bound_95"].extend(lower_bounds.reshape(-1).tolist())
        self.test_outputs["upper_bound_95"].extend(upper_bounds.reshape(-1).tolist())
        self.test_outputs["a"].extend(results["logits"][:, ::2].cpu().numpy().reshape(-1).tolist())
        self.test_outputs["b"].extend(results["logits"][:, 1::2].cpu().numpy().reshape(-1).tolist())
        return results

    def logits_to_bounds(self, logits):
        a = logits[:, ::2].cpu().numpy()
        b = logits[:, 1::2].cpu().numpy()
        return beta.ppf(0.025, a, b), beta.ppf(0.975, a, b)

    def on_test_epoch_start(self) -> None:
        self.test_outputs = {
            "logits": [],
            "prob": [],
            "target": [],
            "histology": [],
            "sample_id": [],
            "split": [],
            "lower_bound_95": [],
            "upper_bound_95": [],
            "a": [],
            "b": [],
        }

    def on_validation_epoch_start(self) -> None:
        pass


def cox_partial_likelihood_loss(risk_scores: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    """Negative Cox partial log-likelihood (Breslow approximation).

    Inputs must be float32 — call inside torch.amp.autocast(enabled=False).

    Args:
        risk_scores: predicted risk scores, shape (batch_size,)
        times: observed times, shape (batch_size,)
        events: event indicators (1=event, 0=censored), shape (batch_size,)

    Returns:
        Scalar loss (negative partial log-likelihood, averaged over events).
    """
    # Sort by descending time so the risk set at each event is a prefix
    order = torch.argsort(times, descending=True)
    risk_scores = risk_scores[order]
    events = events[order]

    # Log-cumulative-sum-exp of risk scores (risk set shrinks as time increases)
    log_cum_hazard = torch.logcumsumexp(risk_scores, dim=0)

    # Only event samples contribute to the loss
    uncensored_mask = events == 1
    if uncensored_mask.sum() == 0:
        # Return zero loss that stays connected to the computation graph
        # so AMP GradScaler can still record inf checks for this optimizer step.
        return 0.0 * risk_scores.sum()

    loss = -torch.mean(risk_scores[uncensored_mask] - log_cum_hazard[uncensored_mask])
    return loss


class JointCoxLightningModule(AbstractJointLightningModule):
    """Lightning module for survival analysis using Cox proportional hazards loss.

    The target tensor is expected to have shape (batch_size, 2 * n_surv_tasks) where
    columns alternate: [time_1, event_1, time_2, event_2, ...].
    """

    logger: NNLogger

    def __init__(self, model, metadata, *args, **kwargs):
        super().__init__(metadata, *args, **kwargs)
        self.calibrator = None

        assert metadata.n_surv_tasks > 0, "JointCoxLightningModule requires at least one survival task"
        assert metadata.n_clf_tasks == 0, "JointCoxLightningModule does not support classification tasks"
        assert metadata.n_reg_tasks == 0, "JointCoxLightningModule does not support regression tasks"

        self.model = hydra.utils.instantiate(
            model,
            num_targets=metadata.n_surv_tasks,  # one risk score per survival task
            _recursive_=False,
        )

    def initialize_metrics(self) -> torchmetrics.MetricCollection:
        metrics = {}
        for split in ["train", "val", "test"]:
            metrics[f"{split}_loss"] = torchmetrics.aggregation.MeanMetric()
        return torchmetrics.MetricCollection(metrics)

    def calculate_loss(self, logits, gt_y, split):
        """Compute Cox partial likelihood loss across survival tasks.

        gt_y has shape (batch_size, 2 * n_surv_tasks): [time_1, event_1, time_2, event_2, ...]
        logits has shape (batch_size, n_surv_tasks): one risk score per task.
        """
        # Disable autocast: logcumsumexp backward is not implemented for Half,
        # and manual .float() casts break GradScaler inf-checking.
        with torch.amp.autocast("cuda", enabled=False):
            logits = logits.float()
            gt_y = gt_y.float()
            total_loss = torch.tensor(0.0, device=logits.device)
            for i in range(self.metadata.n_surv_tasks):
                times = gt_y[:, 2 * i]
                events = gt_y[:, 2 * i + 1]
                risk_scores = logits[:, i]
                total_loss = total_loss + cox_partial_likelihood_loss(risk_scores, times, events)
            return total_loss / max(self.metadata.n_surv_tasks, 1)

    def log_metrics(self, logits, gt_y, loss_value, batch_size, split):
        self.metrics[f"{split}_loss"].update(loss_value)
        self.log_dict({f"loss/{split}": loss_value}, on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)

    def validation_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        results = self._step(batch=batch, split="val")
        logits = results["logits"]

        self.validation_outputs["logits"].extend(logits.detach().cpu().numpy().tolist())
        self.validation_outputs["target"].extend(batch[self.hparams.y_key].cpu().numpy().tolist())
        self._collect_base_outputs(self.validation_outputs, batch, logits)
        return results

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Mapping[str, Any]:
        results = self._step(batch=batch, split="test")
        logits = results["logits"]

        self.test_outputs["logits"].extend(logits.detach().cpu().numpy().tolist())
        self.test_outputs["target"].extend(batch[self.hparams.y_key].cpu().numpy().tolist())
        self._collect_base_outputs(self.test_outputs, batch, logits)
        return {"logits": logits.detach().cpu().numpy(), "loss": results["loss"]}

    def on_test_epoch_start(self) -> None:
        self.test_outputs = {"logits": [], "target": [], "histology": [], "sample_id": [], "split": []}

    def on_validation_epoch_start(self) -> None:
        self.validation_outputs = {"logits": [], "target": [], "histology": [], "sample_id": [], "split": []}
