import ast
import logging
from functools import cached_property, partial
from pathlib import Path
from typing import Dict, Optional

import hydra
import lightning.pytorch as pl
import omegaconf
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, DataLoader, Dataset, IterableDataset
from tqdm import tqdm

from nn_core.common import PROJECT_ROOT

from paladin.data.functional import add_class_weights, collate_fn

pylogger = logging.getLogger(__name__)


class MetaData:
    def __init__(self, target_dict: Dict, longest_sequence: int):
        """The data information the Lightning Module will be provided with.

        This is a "bridge" between the Lightning DataModule and the Lightning Module.
        There is no constraint on the class name nor in the stored information, as long as it exposes the
        `save` and `load` methods.

        The Lightning Module will receive an instance of MetaData when instantiated,
        both in the train loop or when restored from a checkpoint.

        This decoupling allows the architecture to be parametric (e.g. in the number of classes) and
        DataModule/Trainer independent (useful in prediction scenarios).
        MetaData should contain all the information needed at test time, derived from its train dataset.

        Examples are the class names in a classification task or the vocabulary in NLP tasks.
        MetaData exposes `save` and `load`. Those are two user-defined methods that specify
        how to serialize and de-serialize the information contained in its attributes.
        This is needed for the checkpointing restore to work properly.

        Args:
            target_dict: [{"task": "classification", "target": "TP53", "target_type": "gene"}, ...]
            longest_sequence: the longest sequence in the dataset
        """
        self.target_dict: Dict = target_dict
        self.longest_sequence: int = longest_sequence
        print(type(self.target_dict))
        self.n_clf_tasks = len([x for x in self.target_dict["task"] if x == "classification"])
        self.n_reg_tasks = len([x for x in self.target_dict["task"] if x == "regression"])
        self.n_surv_tasks = len([x for x in self.target_dict["task"] if x == "survival"])

        self.clf_targets = []
        self.reg_targets = []
        self.surv_targets = []  # each entry: {"time": time_col, "event": event_col}
        for idx in range(len(self.target_dict["target"])):
            if self.target_dict["task"][idx] == "classification":
                self.clf_targets.append(self.target_dict["target"][idx])
            elif self.target_dict["task"][idx] == "regression":
                self.reg_targets.append(self.target_dict["target"][idx])
            elif self.target_dict["task"][idx] == "survival":
                # Survival targets encode "time_col:event_col" in the target name
                parts = self.target_dict["target"][idx].split(":")
                if len(parts) == 2:
                    self.surv_targets.append({"time": parts[0], "event": parts[1]})
                else:
                    raise ValueError(
                        f"Survival target must be formatted as 'time_col:event_col', got: {self.target_dict['target'][idx]}"
                    )
            else:
                raise ValueError(f"Unknown task type: {self.target_dict['task'][idx]}")

    def get_targets(self):
        return self.target_dict["target"]

    def save(self, dst_path: Path) -> None:
        """Serialize the MetaData attributes into the zipped checkpoint in dst_path.

        Args:
            dst_path: the root folder of the metadata inside the zipped checkpoint
        """
        pylogger.debug(f"Saving MetaData to '{dst_path}'")

        (dst_path / "target_dicts.tsv").write_text("\n".join(f"{d}" for d in [self.target_dict]))
        (dst_path / "longest_sequence.txt").write_text(str(self.longest_sequence))

    @staticmethod
    def load(src_path: Path) -> "MetaData":
        """Deserialize the MetaData from the information contained inside the zipped checkpoint in src_path.

        Args:
            src_path: the root folder of the metadata inside the zipped checkpoint

        Returns:
            an instance of MetaData containing the information in the checkpoint
        """
        pylogger.debug(f"Loading MetaData from '{src_path}'")

        lines = (src_path / "target_dicts.tsv").read_text(encoding="utf-8").splitlines()

        target_dicts = []
        for line in lines:
            target_dicts.append(ast.literal_eval(line))
        assert len(target_dicts) == 1, "Expected 1 target_dict, got {len(target_dicts)}"
        target_dict = target_dicts[0]

        longest_sequence = int((src_path / "longest_sequence.txt").read_text(encoding="utf-8"))

        return MetaData(
            target_dict=target_dict,
            longest_sequence=longest_sequence,
        )

    def __repr__(self) -> str:
        attributes = ",\n    ".join([f"{key}={value}" for key, value in self.__dict__.items()])
        return f"{self.__class__.__name__}(\n    {attributes}\n)"


class JointDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        accelerator: str,
    ):
        """Initializes PaladinDataModule.

        Args:
            dataset: the dataset configuration
            num_workers: the number of workers to use in the dataloaders
            batch_size: the batch size configuration
            accelerator: the accelerator type

        Returns:
            None
        """
        super().__init__()
        self.dataset = dataset
        self.num_workers = num_workers
        self.batch_size = batch_size
        # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#gpus
        self.pin_memory: bool = False  # accelerator is not None and str(accelerator) == "gpu"
        print("WARNING: pin_memory is set to False")

        self.datasets: Dict[str, Dataset] = {}

    @cached_property
    def metadata(self) -> MetaData:
        """Data information to be fed to the Lightning Module as parameter.

        Examples are vocabularies, number of classes...

        Returns:
            metadata: everything the model should know about the data, wrapped in a MetaData object.
        """
        if self.datasets["train"] is None:
            self.setup(stage="fit")

        self.datasets["train"].task = add_class_weights(self.datasets["train"])
        print(self.datasets["train"].task)
        longest_sequence = self.datasets["train"].longest_sequence
        print(longest_sequence)
        return MetaData(target_dict=self.datasets["train"].task, longest_sequence=longest_sequence)

    def setup(self, stage: Optional[str] = None):
        self.datasets = hydra.utils.instantiate(self.dataset)

    def on_train_epoch_start(self) -> None:
        """Called at the start of each training epoch to advance IterableDataset shuffling."""
        if self.trainer is None:
            return

        train_ds = self.datasets.get("train")
        if train_ds is not None and isinstance(train_ds, IterableDataset) and hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(self.trainer.current_epoch)

    def on_validation_epoch_start(self) -> None:
        """Called at the start of each validation epoch to ensure deterministic evaluation."""
        if self.trainer is None:
            return

        val_ds = self.datasets.get("val")
        if val_ds is not None and isinstance(val_ds, IterableDataset) and hasattr(val_ds, "set_epoch"):
            val_ds.set_epoch(self.trainer.current_epoch)

    def train_dataloader(self) -> DataLoader:
        train_ds = self.datasets["train"]
        is_iterable = isinstance(train_ds, IterableDataset)
        # 'spawn' avoids CUDA-fork deadlocks (CUDA init in parent → inherited mutex locks in
        # forked children → hang).  'forkserver' also avoids the deadlock but fails with
        # "too many fds" when the parent has many open handles (CUDA, shared memory, etc.).
        # 'spawn' re-imports all modules in each worker; slower first-batch but always safe.
        mp_ctx = "spawn" if is_iterable and self.num_workers.train > 0 else None
        return DataLoader(
            train_ds,
            shuffle=not is_iterable,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            pin_memory=self.pin_memory,
            drop_last=True,  # Always drop last incomplete batch for training
            collate_fn=partial(collate_fn, split="train", metadata=self.metadata),
            multiprocessing_context=mp_ctx,
        )

    def val_dataloader(self) -> DataLoader:
        val_ds = self.datasets["val"]
        is_iterable = isinstance(val_ds, IterableDataset)
        mp_ctx = "spawn" if is_iterable and self.num_workers.val > 0 else None
        return DataLoader(
            val_ds,
            shuffle=False,
            batch_size=self.batch_size.val,
            num_workers=self.num_workers.val,
            pin_memory=self.pin_memory,
            drop_last=False,  # Never drop data for validation
            collate_fn=partial(collate_fn, split="val", metadata=self.metadata),
            multiprocessing_context=mp_ctx,
        )

    def test_dataloader(self) -> DataLoader:
        ds_list = []
        ds_values = list(self.datasets.keys())
        if "test" in ds_values:
            ds_list.append(self.datasets["test"])
            ds_values.remove("test")
        for ds_key in ds_values:
            ds_list.append(self.datasets[ds_key])

        # IterableDatasets cannot be used with ConcatDataset; use ChainDataset instead
        is_iterable = any(isinstance(ds, IterableDataset) for ds in ds_list)
        if is_iterable:
            from torch.utils.data import ChainDataset

            ds = ChainDataset(ds_list)
        else:
            ds = ConcatDataset(ds_list)

        mp_ctx = "spawn" if is_iterable and self.num_workers.test > 0 else None
        return [
            DataLoader(
                ds,
                shuffle=False,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                pin_memory=self.pin_memory,
                drop_last=False,  # Never drop data for testing
                collate_fn=partial(collate_fn, split="test", metadata=self.metadata),
                multiprocessing_context=mp_ctx,
            )
        ]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" f"{self.dataset=}, " f"{self.num_workers=}, " f"{self.batch_size=})"  # type: ignore


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base=None)
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the DataModule.

    Args:
        cfg: the hydra configuration
    """
    import os
    import tempfile
    scratch_dir = Path(os.getenv("PALADIN_DEBUG_SCRATCH_DIR", tempfile.gettempdir())) / "paladin_scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    
    m: pl.LightningDataModule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)
    m.metadata
    m.metadata.save(scratch_dir)
    m.metadata.load(scratch_dir)
    m.setup()

    for batch in tqdm(m.train_dataloader()):
        # print(batch["image_ids"])
        print(batch["tile_tensor"].shape)
        print(batch["target"])
        print(batch["site"])
        print(batch["oncotree_code"])
        print(batch["histologic_embedding"].shape)
        print(batch["target_embedding"].shape)


if __name__ == "__main__":
    main()
