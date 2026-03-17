import os
import json
from pathlib import Path

import hydra
import numpy as np
import omegaconf
import pandas as pd
import torch
import wandb
from lifelines.utils import concordance_index
from lightning.pytorch.callbacks import Callback
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from nn_core.common import PROJECT_ROOT

import paladin.utils.stats
from paladin.modules.module import AeonLateAggregator


def bootstrap_concordance_ci(risk_scores, times, events, n_bootstraps=1000, ci=0.95):
    """Bootstrap confidence interval for the concordance index.

    Uses lifelines.utils.concordance_index internally.

    Returns:
        tuple: (c_index, lower, upper)
    """
    risk_scores = np.asarray(risk_scores)
    times = np.asarray(times)
    events = np.asarray(events)
    n = len(risk_scores)
    c_values = []
    for _ in range(n_bootstraps):
        idx = np.random.choice(n, n, replace=True)
        c = concordance_index(times[idx], -risk_scores[idx], events[idx])
        if not np.isnan(c):
            c_values.append(c)
    if len(c_values) < 0.9 * n_bootstraps:
        return np.nan, np.nan, np.nan
    alpha = (1 - ci) / 2
    c_index_val = concordance_index(times, -risk_scores, events)
    return c_index_val, np.percentile(c_values, 100 * alpha), np.percentile(c_values, 100 * (1 - alpha))

# adjust to run only on clf targets and to store for multiple targets


class ConvertToONNX(Callback):
    def __init__(self, export_path):
        self.export_path = export_path

    def on_test_epoch_start(self, trainer, pl_module):
        print("on_test_epoch_start of ConvertToONNX callback running")
        # input_sample = torch.randn(1, 20000, 1536).float()
        # print(f"ONNX model saved to {self.export_path}")
        # pl_module.to_onnx(self.export_path, input_sample, export_params=True, dynamo=True)
        # print(f"ONNX model saved to {self.export_path}")
        import pickle  # nosec

        with open(self.export_path.replace(".onnx", ".pkl"), "wb") as f:
            pickle.dump(pl_module.model.cpu(), f)  # nosec
        exit()


class CalibrationCallback(Callback):
    def on_fit_end(self, trainer, pl_module):
        if pl_module.metadata.n_clf_tasks > 1 or pl_module.metadata.n_reg_tasks > 0:
            print(
                "Skipping calibration callback because there are multiple classification tasks and/or a non-zero number of regression tasks"
            )
        elif trainer.checkpoint_callback.best_model_path is not None:
            trainer.model.eval()
            trainer.validate(model=trainer.model, datamodule=trainer.datamodule, ckpt_path="best")
            calibrator = self.generate_calibrator(pl_module.validation_outputs)
            pl_module.calibrator = calibrator
            trainer.save_checkpoint(trainer.checkpoint_callback.best_model_path, weights_only=False)
        else:
            print("No best model found")

    @staticmethod
    def generate_calibrator(validation_outputs):
        # print(validation_outputs["logits"])
        # print(type(validation_outputs["logits"]))
        X = np.array(validation_outputs["logits"]).reshape(-1, 1)
        y = np.array(validation_outputs["target"]).ravel()
        clf = LogisticRegression(C=1e-4)
        cccv = CalibratedClassifierCV(clf, cv=5, method="sigmoid")
        cccv.fit(X, y)
        return cccv


class TestMetricsCallback(Callback):
    def on_test_epoch_end(self, trainer, pl_module):
        print("on_test_epoch_end running")
        print(pl_module.test_outputs)

        n_clf_tasks = pl_module.metadata.n_clf_tasks
        n_reg_tasks = pl_module.metadata.n_reg_tasks
        total_tasks = n_clf_tasks + n_reg_tasks
        print(f"n_clf_tasks: {n_clf_tasks}")
        print(f"n_reg_tasks: {n_reg_tasks}")
        print(f"total_tasks: {total_tasks}")

        # Convert lists to numpy arrays
        logits = np.array(pl_module.test_outputs["logits"])
        probs = np.array(pl_module.test_outputs["prob"])
        targets = np.array(pl_module.test_outputs["target"])
        histologies = np.array(pl_module.test_outputs["histology"])
        sample_ids = np.array(pl_module.test_outputs["sample_id"])
        splits = np.array(pl_module.test_outputs["split"])

        # Ensure that the total number of logits and targets align with the number of tasks
        if logits.shape[0] % total_tasks != 0:
            raise ValueError("Logits size is not divisible by the total number of tasks.")
        if targets.shape[0] % total_tasks != 0:
            raise ValueError("Targets size is not divisible by the total number of tasks.")
        if probs.shape[0] % total_tasks != 0:
            raise ValueError("Probabilities size is not divisible by the total number of tasks.")

        num_samples = logits.shape[0] // total_tasks

        # Reshape to [num_samples, total_tasks]
        logits = logits.reshape(num_samples, total_tasks)
        probs = probs.reshape(num_samples, total_tasks)
        targets = targets.reshape(num_samples, total_tasks)
        histologies = histologies.reshape(num_samples, total_tasks)
        sample_ids = sample_ids.reshape(num_samples, total_tasks)
        splits = splits.reshape(num_samples, total_tasks)
        for row_idx in range(num_samples):
            assert np.unique(splits[row_idx]).size == 1, f"splits are not the same for row_idx: {row_idx}"

        all_data = []

        # Get target names from metadata
        clf_targets = (
            pl_module.metadata.clf_targets
            if hasattr(pl_module.metadata, "clf_targets")
            else [f"clf_{i}" for i in range(n_clf_tasks)]
        )
        reg_targets = (
            pl_module.metadata.reg_targets
            if hasattr(pl_module.metadata, "reg_targets")
            else [f"reg_{i}" for i in range(n_reg_tasks)]
        )
        print(f"clf_targets: {clf_targets}")
        print(f"reg_targets: {reg_targets}")

        # Iterate over each task
        for task_idx in range(total_tasks):
            print(f"iterating over task_idx: {task_idx}")
            if task_idx < n_clf_tasks:
                task_type = "classification"
                target = targets[:, task_idx]
                logit = logits[:, task_idx]
                prob = probs[:, task_idx]
                target_name = clf_targets[task_idx]
            else:
                task_type = "regression"
                reg_task_idx = task_idx - n_clf_tasks
                target = targets[:, task_idx]
                logit = logits[:, task_idx]
                prob = probs[:, task_idx]
                target_name = reg_targets[reg_task_idx]

            task_df = pd.DataFrame(
                {
                    "task_type": task_type,
                    "task_idx": task_idx if task_type == "classification" else reg_task_idx,
                    "target_name": target_name,
                    "histology": histologies[:, task_idx],
                    "target": target,
                    "logit": logit,
                    "prob": prob,
                    "sample_id": sample_ids[:, task_idx],
                    "split": splits[:, task_idx],
                }
            )
            all_data.append(task_df)

        # Combine all task data into a single DataFrame
        combined_df = pd.concat(all_data, ignore_index=True)
        assert (
            len(combined_df.target_name.unique()) == total_tasks
        ), f"Number of unique targets does not match the total number of tasks: {len(combined_df.target.unique())} != {total_tasks}"

        # Log the combined DataFrame as an artifact
        artifact = wandb.Artifact(name="inference_df", type="df")
        table = wandb.Table(dataframe=combined_df)
        artifact.add(table, "inference_df")
        wandb.log_artifact(artifact)

        # Filter only test split data
        test_df = combined_df[combined_df["split"] == "test"]
        del combined_df
        assert "val" not in test_df["split"].unique(), "Validation data found in test data"
        assert "train" not in test_df["split"].unique(), "Train data found in test data"

        # Create consolidated results for classification and regression tasks
        clf_results = []
        reg_results = []

        # Iterate over each task and collect metrics
        for (task_type, task_idx, target_name), group in test_df.groupby(["task_type", "task_idx", "target_name"]):
            if task_type == "classification":
                results = paladin.utils.stats.create_binary_auroc_table(group)["df"]
                results["target_name"] = target_name  # add column for target name
                assert "target_name" in results.columns, "target_name column not found in results"
                clf_results.append(results)
                # Still log individual calibration plots
                calibration_plot = paladin.utils.stats.plot_calibration_curve(group)
                wandb.log({f"test_calibration_plot_{target_name}": calibration_plot})
            elif task_type == "regression":
                results = paladin.utils.stats.create_pearson_table(group)["df"]
                results["target_name"] = target_name  # add column for target name
                assert "target_name" in results.columns, "target_name column not found in results"
                reg_results.append(results)
            else:
                print(f"Unsupported task type: {task_type}")

        # Log consolidated tables if there are results
        if clf_results:
            combined_clf_df = pd.concat(clf_results)
            # Combine classification results into one table
            combined_clf_table = wandb.Table(dataframe=combined_clf_df)
            wandb.log({"test_classification_table": combined_clf_table})
            wandb.log({"auroc/test": combined_clf_df["auroc"].dropna().mean()})

        if reg_results:
            combined_reg_df = pd.concat(reg_results)
            # Combine regression results into one table
            combined_reg_table = wandb.Table(dataframe=combined_reg_df)
            wandb.log({"test_regression_table": combined_reg_table})
            wandb.log({"pearson/test": combined_reg_df["pearson"].dropna().mean()})


class BetaBinomialTestMetricsCallback(Callback):
    def on_test_epoch_end(self, trainer, pl_module):
        print("on_test_epoch_end running")

        # Convert lists to numpy arrays
        logits = np.array(pl_module.test_outputs["logits"]).reshape(-1)
        lower_bounds = np.array(pl_module.test_outputs["lower_bound_95"])
        upper_bounds = np.array(pl_module.test_outputs["upper_bound_95"])
        targets = np.array(pl_module.test_outputs["target"])
        histologies = np.array(pl_module.test_outputs["histology"])
        sample_ids = np.array(pl_module.test_outputs["sample_id"])
        splits = np.array(pl_module.test_outputs["split"])
        a = np.array(pl_module.test_outputs["a"])
        b = np.array(pl_module.test_outputs["b"])

        assert (
            len(pl_module.metadata.clf_targets) == 1
        ), "Only one classification target is supported for beta-binomial loss"
        assert len(pl_module.metadata.reg_targets) == 0, "No regression targets are supported for beta-binomial loss"
        clf_target = pl_module.metadata.clf_targets[0]
        print(f"clf_target: {clf_target}\n")

        task_df = pd.DataFrame(
            {
                "task_type": "classification",
                "target_name": [clf_target] * len(logits),
                "histology": histologies,
                "target": targets,
                "logit": logits,
                "prob": logits,
                "lower_bound_95": lower_bounds,
                "upper_bound_95": upper_bounds,
                "a": a,
                "b": b,
                "sample_id": sample_ids,
                "split": splits,
            }
        )

        # Log the DataFrame as an artifact
        artifact = wandb.Artifact(name="inference_df", type="df")
        table = wandb.Table(dataframe=task_df)
        artifact.add(table, "inference_df")
        wandb.log_artifact(artifact)

        self.single_split_metrics(task_df, "test", clf_target)
        self.single_split_metrics(task_df, "tcga", clf_target)

    def single_split_metrics(self, task_df, split, clf_target):
        # Filter only test split data
        test_df = task_df[task_df["split"] == split].copy()
        if len(test_df) == 0:
            print(f"No data found for split: {split}")
            return
        unique_splits = test_df.split.unique()
        assert len(unique_splits) == 1, f"More than {split} split found in data, specifically: {unique_splits}"

        # Iterate over each task and collect metrics
        clf_results = paladin.utils.stats.create_binary_auroc_table(test_df)["df"]
        clf_results["target_name"] = clf_target  # add column for target name
        combined_clf_table = wandb.Table(dataframe=clf_results)
        wandb.log({f"{split}_classification_table": combined_clf_table})
        wandb.log({f"auroc/{split}": clf_results["auroc"].dropna().mean()})

        # Still log individual calibration plots
        calibration_plot = paladin.utils.stats.plot_calibration_curve(test_df)
        wandb.log({f"{split}_calibration_plot_{clf_target}": calibration_plot})

        # plot decision curve
        decision_curve = paladin.utils.stats.plot_decision_curve(test_df)
        wandb.log({f"{split}_decision_curve_{clf_target}": decision_curve})


class BetaBinomialMetaTestMetricsCallback(Callback):
    def on_test_epoch_end(self, trainer, pl_module):
        print("on_test_epoch_end running")

        # Convert lists to numpy arrays
        logits = np.array(pl_module.test_outputs["logits"]).reshape(-1)
        # lower_bounds = np.array(pl_module.test_outputs["lower_bound_95"])
        # upper_bounds = np.array(pl_module.test_outputs["upper_bound_95"])
        targets = np.array(pl_module.test_outputs["target"])
        histologies = np.array(pl_module.test_outputs["histology"])
        sample_ids = np.array(pl_module.test_outputs["sample_id"])
        splits = np.array(pl_module.test_outputs["split"])
        # a = np.array(pl_module.test_outputs["a"])
        # b = np.array(pl_module.test_outputs["b"])
        target_names = np.array(pl_module.test_outputs["target_name"])

        task_df = pd.DataFrame(
            {
                "task_type": "classification",
                "histology": histologies,
                "target": targets,
                "logit": logits,
                "prob": logits,
                # "lower_bound_95": lower_bounds,
                # "upper_bound_95": upper_bounds,
                # "a": a,
                # "b": b,
                "sample_id": sample_ids,
                "split": splits,
                "target_name": target_names,
            }
        )

        print(f"task_df: {task_df}")

        # Log the DataFrame as an artifact
        print("logging task_df to wandb")
        artifact = wandb.Artifact(name="inference_df", type="df")
        table = wandb.Table(dataframe=task_df)
        artifact.add(table, "inference_df")
        wandb.log_artifact(artifact)

        print("logging task_df to wandb complete")

        for clf_target in task_df.target_name.unique():
            for histology in task_df.histology.unique():
                print(f"calculating metrics for target: {clf_target} and histology: {histology}")
                self.single_split_metrics(
                    task_df[(task_df["target_name"] == clf_target) & (task_df["histology"] == histology)],
                    "test",
                    clf_target,
                    histology,
                )
                self.single_split_metrics(
                    task_df[(task_df["target_name"] == clf_target) & (task_df["histology"] == histology)],
                    "tcga",
                    clf_target,
                    histology,
                )

    def single_split_metrics(self, test_df, split, clf_target, histology):
        # Iterate over each task and collect metrics
        clf_results = paladin.utils.stats.create_binary_auroc_table(test_df)["df"]
        combined_clf_table = wandb.Table(dataframe=clf_results)
        wandb.log({f"{split}_classification_table_{clf_target}_{histology}": combined_clf_table})
        wandb.log({f"auroc/{split}_{clf_target}_{histology}": clf_results["auroc"].dropna().mean()})

        # Still log individual calibration plots
        try:
            calibration_plot = paladin.utils.stats.plot_calibration_curve(test_df)
            wandb.log({f"{split}_calibration_plot_{clf_target}_{histology}": calibration_plot})
        except Exception as e:
            print(f"Error logging calibration plot for target {clf_target} and histology {histology}: {e}")

        # plot decision curve
        try:
            decision_curve = paladin.utils.stats.plot_decision_curve(test_df)
            wandb.log({f"{split}_decision_curve_{clf_target}_{histology}": decision_curve})
        except Exception as e:
            print(f"Error logging decision curve for target {clf_target} and histology {histology}: {e}")


class CoxTestMetricsCallback(Callback):
    """Callback for survival tasks: computes concordance index with bootstrapped CIs on train/val/test."""

    def on_test_epoch_end(self, trainer, pl_module):
        print("CoxTestMetricsCallback on_test_epoch_end running")
        n_surv_tasks = pl_module.metadata.n_surv_tasks
        surv_targets = pl_module.metadata.surv_targets

        logits = np.array(pl_module.test_outputs["logits"])  # (N, n_surv_tasks)
        targets = np.array(pl_module.test_outputs["target"])  # (N, 2 * n_surv_tasks)
        histologies = np.array(pl_module.test_outputs["histology"])
        sample_ids = np.array(pl_module.test_outputs["sample_id"])
        splits = np.array(pl_module.test_outputs["split"])

        all_data = []
        for task_idx in range(n_surv_tasks):
            surv_info = surv_targets[task_idx]
            target_name = f"{surv_info['time']}:{surv_info['event']}"
            risk_scores = logits[:, task_idx]
            times = targets[:, 2 * task_idx]
            events = targets[:, 2 * task_idx + 1]
            task_df = pd.DataFrame(
                {
                    "task_type": "survival",
                    "target_name": target_name,
                    "histology": histologies,
                    "risk_score": risk_scores,
                    "time": times,
                    "event": events,
                    "sample_id": sample_ids,
                    "split": splits,
                }
            )
            all_data.append(task_df)

        combined_df = pd.concat(all_data, ignore_index=True)

        # Log full inference dataframe as artifact
        artifact = wandb.Artifact(name="inference_df", type="df")
        table = wandb.Table(dataframe=combined_df)
        artifact.add(table, "inference_df")
        wandb.log_artifact(artifact)

        # Compute c-index per split and task
        for split_name in combined_df["split"].unique():
            split_df = combined_df[combined_df["split"] == split_name]
            surv_results = []
            for target_name, group in split_df.groupby("target_name"):
                for hist, hist_group in group.groupby("histology"):
                    c_idx, c_lower, c_upper = bootstrap_concordance_ci(
                        hist_group["risk_score"].values,
                        hist_group["time"].values,
                        hist_group["event"].values,
                    )
                    surv_results.append(
                        {
                            "target_name": target_name,
                            "histology": hist,
                            "c_index": c_idx,
                            "c_index_lower_95": c_lower,
                            "c_index_upper_95": c_upper,
                            "n": len(hist_group),
                            "n_events": int(hist_group["event"].sum()),
                        }
                    )
            if surv_results:
                surv_results_df = pd.DataFrame(surv_results)
                wandb.log({f"{split_name}_survival_table": wandb.Table(dataframe=surv_results_df)})
                wandb.log({f"c_index/{split_name}": surv_results_df["c_index"].dropna().mean()})

    def on_validation_epoch_end(self, trainer, pl_module):
        """Compute and log c-index at end of each validation epoch for monitoring."""
        if not hasattr(pl_module, "validation_outputs") or len(pl_module.validation_outputs.get("logits", [])) == 0:
            return
        n_surv_tasks = pl_module.metadata.n_surv_tasks
        logits = np.array(pl_module.validation_outputs["logits"])
        targets = np.array(pl_module.validation_outputs["target"])

        c_indices = []
        for task_idx in range(n_surv_tasks):
            risk_scores = logits[:, task_idx]
            times = targets[:, 2 * task_idx]
            events = targets[:, 2 * task_idx + 1]
            c_idx = concordance_index(times, -risk_scores, events)
            c_indices.append(c_idx)

        mean_c = np.nanmean(c_indices)
        pl_module.log("c_index/val", mean_c, on_epoch=True, sync_dist=True)


class SaveTissueSiteEmbeddingsCallback(Callback):
    def __init__(self, output_path: str = "tissue_site_embeddings.json"):
        self.output_path = output_path

    def on_test_epoch_start(self, trainer, pl_module):
        if isinstance(pl_module.model, AeonLateAggregator):
            embeddings = pl_module.model.tissue_site_projector.weight.detach().cpu().numpy()
            embeddings_dict = {str(i): embeddings[i].tolist() for i in range(len(embeddings))}
            with open(self.output_path, "w") as f:
                json.dump(embeddings_dict, f)
        exit()


class TransferLearningCallback(Callback):
    def __init__(self, weights_path, freeze_layers=False):
        super().__init__()
        self.weights_path = weights_path
        self.freeze_layers = freeze_layers

    def on_fit_start(self, trainer, pl_module):
        if not self.weights_path or not os.path.exists(self.weights_path):
            print(f"TransferLearningCallback: weights_path not found, skipping: {self.weights_path}")
            return
        print(f"loading from weights_path: {self.weights_path}")
        state_dict = torch.load(self.weights_path, weights_only=True, map_location=pl_module.device)
        for key in state_dict.keys():
            if key in pl_module.model.state_dict().keys():
                pl_module.model.state_dict()[key] = state_dict[key]
            else:
                print(f"key {key} not found in model state dict")

        if self.freeze_layers:
            for param in pl_module.model.parameters():
                param.requires_grad = False
            pl_module.model.eval()


class ExportCallback(Callback):
    def setup(self, trainer, pl_module, stage):
        pl_module.outputs = {"train": [], "val": [], "test": []}

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        self.save_forward_updates(pl_module, "train")

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self.save_forward_updates(pl_module, "val")

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        self.save_forward_updates(pl_module, "test")

    def save_forward_updates(self, pl_module, split):
        output_dict = pl_module.outputs[f"{split}"]
        df = pd.DataFrame(output_dict)

        df_logits = pd.DataFrame(df["logits"].to_list(), index=df.index)
        df_logits.columns = pl_module.metadata.get_targets()

        df = pd.concat([df.drop(columns=["logits"]), df_logits.add_prefix("pred_")], axis=1)

        path = Path(f"{pl_module.logger.run_dir}/{split}")
        path.mkdir(parents=True, exist_ok=True)

        # output is at model.run_dir/{split}/epoch={epoch}-step={step}.outputs.parquet
        df.to_parquet(
            f"{pl_module.logger.run_dir}/{split}/epoch={pl_module.current_epoch}-step={pl_module.global_step}.outputs.parquet"
        )

        pl_module.outputs[f"{split}"] = []

    @staticmethod
    def update_forward_outputs(pl_module, batch, results, split):
        sample_id = batch["sample_id"]
        image_ids = batch["image_ids"]
        split_batch = batch["split"]

        for idx, _ in enumerate(sample_id):
            output = {
                "sample_id": sample_id[idx],
                "image_ids": image_ids[idx],
                "split_batch": split_batch[idx],
            }

            for key in results.keys():
                output[key] = results[key][idx].detach().cpu().to(torch.float32).flatten().numpy()

            pl_module.outputs[f"{split}"].append(output)

        print(f"Updated forward outputs for [{split}], length: {len(pl_module.outputs[f'{split}'])}")


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base=None)
def main_generate_weights(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the DataModule.

    Args:
        cfg: the hydra configuration
    """
    data_module = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)
    metadata = data_module.metadata
    module = hydra.utils.instantiate(
        cfg.nn.module,
        _recursive_=False,
        metadata=metadata,
        ckpt_path="storage/paladin/x4cyethl/checkpoints/epoch=10-step=308.ckpt.zip",
    )
    torch.save(module.model.state_dict(), "transfer-learning-weights-x4cyethl-epoch-10-step-308.pth")
