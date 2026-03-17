import logging
import os
from typing import Dict, List, Optional

import hydra
import lightning.pytorch as pl
import omegaconf
import torch
import wandb
from lightning.pytorch import Callback
from lightning.pytorch.tuner import Tuner
from omegaconf import DictConfig, ListConfig, OmegaConf

from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import seed_index_everything
from nn_core.model_logging import NNLogger
from nn_core.serialization import NNCheckpointIO

# Force the execution of __init__.py if this file is executed directly.
import paladin  # noqa
from paladin.data.joint_datamodule import MetaData

pylogger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


def build_callbacks(cfg: ListConfig, *args: Callback) -> List[Callback]:
    """Instantiate the callbacks given their configuration.

    Args:
        cfg: a list of callbacks instantiable configuration
        *args: a list of extra callbacks already instantiated

    Returns:
        the complete list of callbacks to use
    """
    callbacks: List[Callback] = list(args)

    for callback in cfg:
        pylogger.info(f"Adding callback <{callback['_target_'].split('.')[-1]}>")
        callbacks.append(hydra.utils.instantiate(callback, _recursive_=False))

    return callbacks


def run(cfg: DictConfig) -> str:
    """Generic train loop.

    Args:
        cfg: run configuration, defined by Hydra in /conf

    Returns:
        the run directory inside the storage_dir used by the current experiment
    """
    print(cfg)

    seed_index_everything(cfg.train)
    try:
        with open("wandb.key", "r") as file:
            key = file.read().replace("\n", "")
        wandb.login(key=key)
    except FileNotFoundError:
        pylogger.warning("wandb.key not found — falling back to offline mode. Create a wandb.key file with your API key to enable online logging.")
        os.environ["WANDB_MODE"] = "offline"

    fast_dev_run: bool = cfg.train.trainer.fast_dev_run
    if fast_dev_run:
        pylogger.info(f"Debug mode <{cfg.train.trainer.fast_dev_run=}>. Forcing debugger friendly configuration!")
        # Debuggers don't like GPUs nor multiprocessing, but alas fast attention is not supported on CPU
        cfg.train.trainer.accelerator = "gpu"
        cfg.train.trainer.devices = 1
        cfg.nn.data.num_workers.train = 0
        cfg.nn.data.num_workers.val = 0
        cfg.nn.data.num_workers.test = 0
        cfg.nn.data.batch_size.train = 1

    # cfg.core.tags = enforce_tags(cfg.core.get("tags", None))

    # Instantiate datamodule
    pylogger.info(f"Instantiating <{cfg.nn.data['_target_']}>")
    print(f"Instantiating <{cfg.nn.data['_target_']}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)
    print(f"Instantiated <{datamodule.__class__.__name__}>")
    datamodule.setup(stage=None)
    print(f"Set up <{datamodule.__class__.__name__}>")

    metadata: Optional[MetaData] = getattr(datamodule, "metadata", None)
    if metadata is None:
        pylogger.warning(f"No 'metadata' attribute found in datamodule <{datamodule.__class__.__name__}>")

    # Instantiate model
    pylogger.info(f"Instantiating <{cfg.nn.module['_target_']}>")
    print(f"Instantiating <{cfg.nn.module['_target_']}>")
    model: pl.LightningModule = hydra.utils.instantiate(cfg.nn.module, _recursive_=False, metadata=metadata)
    print(model)

    # Instantiate the callbacks
    print("instantiating callbacks")
    template_core: NNTemplateCore = NNTemplateCore(
        restore_cfg=cfg.train.get("restore", None),
    )
    callbacks: List[Callback] = build_callbacks(cfg.train.callbacks, template_core)

    storage_dir: str = cfg.core.storage_dir

    logger: NNLogger = NNLogger(logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id)

    # Extract hash - run_dir is in format storage/{project_name}/{hash}
    run_hash = logger.experiment.id
    # if hasattr(logger.experiment, "name"):
    #     logger.experiment.name = f"{logger.experiment.name}_{run_hash}"

    # Add to model for tracking
    model.run_hash = run_hash
    model.storage_path = logger.run_dir

    pylogger.info("Instantiating the <Trainer>")
    trainer = pl.Trainer(
        default_root_dir=storage_dir,
        plugins=[NNCheckpointIO(jailing_dir=logger.run_dir)],
        logger=logger,
        callbacks=callbacks,
        **cfg.train.trainer,
    )

    if cfg.train.command == "tune":
        tuner = Tuner(trainer)
        tuner.lr_find(model, datamodule=datamodule, max_lr=1e-2)
    elif cfg.train.command == "run":
        pylogger.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=template_core.trainer_ckpt_path)

        if fast_dev_run:
            pylogger.info("Skipping testing in 'fast_dev_run' mode!")
        else:
            if trainer.checkpoint_callback.best_model_path is not None:
                pylogger.info("Starting testing!")
                trainer.test(datamodule=datamodule)
    elif cfg.train.command == "test":
        pylogger.info(f"Starting testing! with {template_core.trainer_ckpt_path}")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=template_core.trainer_ckpt_path)
    else:
        raise ValueError(f"Unknown command: {cfg.train.command}")

    # Logger closing to release resources/avoid multi-run conflicts
    if logger is not None:
        logger.experiment.finish()

    return logger.run_dir


def _get_task_field_string(tasks: List[Dict], field: str, fallback: str) -> str:
    try:
        unique_values = list(set(v for x in tasks for v in x[field]))
        return "-".join(unique_values)
    except:  # noqa
        return fallback


def get_histstring(tasks: List[Dict]) -> str:
    return _get_task_field_string(tasks, "histologies", "allhists")


def get_sitestring(tasks: List[Dict]) -> str:
    return _get_task_field_string(tasks, "sites", "allsites")


def get_targetstring(tasks: List[Dict]) -> str:
    assert len(tasks) == 1, "Only one task is supported for target string"
    targets = list(set([x for x in tasks[0]["target"]]))
    if len(targets) <= 3:
        return "-".join(targets)
    else:
        return f"multitask{len(targets)}"


# @hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.1")
@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base=None)
def main(cfg: omegaconf.DictConfig):
    OmegaConf.register_new_resolver("taskstring", get_targetstring)
    OmegaConf.register_new_resolver("histstring", get_histstring)
    OmegaConf.register_new_resolver("sitestring", get_sitestring)
    run(cfg)


if __name__ == "__main__":
    main()
