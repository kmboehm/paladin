import ast
import logging
from functools import cached_property
from pathlib import Path
from typing import Dict, List

import hydra
import lightning.pytorch as pl
import omegaconf
from omegaconf import DictConfig
from tqdm import tqdm

from nn_core.common import PROJECT_ROOT

from paladin.data.functional import add_class_weights_multiclass
from paladin.data.joint_datamodule import JointDataModule

pylogger = logging.getLogger(__name__)


class MetaData:
    def __init__(
        self,
        target_dicts: List[Dict],
        longest_sequence: int,
        int_to_name_class_mapping: Dict,
        n_classes: int,
        ontology_embedding_dim: int,
    ):
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
            target_dicts: [{"task": "classification", "target": "TP53", "target_type": "gene"}, ...]
            longest_sequence: the longest sequence in the dataset
            int_to_name_class_mapping: a dictionary mapping the integer class to the class name
            n_classes: the number of classes
            ontology_embedding_dim: the dimension of the ontology embedding
        """
        self.target_dicts: List[Dict] = target_dicts
        assert len(self.target_dicts) == 1, "This module is only supported for single-task multiclass classification"
        self.longest_sequence: int = longest_sequence
        self.n_classes = n_classes
        self.int_to_name_class_mapping = int_to_name_class_mapping
        self.ontology_embedding_dim = ontology_embedding_dim

    def get_targets(self):
        class_names = [self.int_to_name_class_mapping[key] for key in sorted(self.int_to_name_class_mapping.keys())]
        return class_names
    
    def save(self, dst_path: Path) -> None:
        """Serialize the MetaData attributes into the zipped checkpoint in dst_path.

        Args:
            dst_path: the root folder of the metadata inside the zipped checkpoint
        """
        pylogger.debug(f"Saving MetaData to '{dst_path}'")

        (dst_path / "target_dict.tsv").write_text(str(self.target_dicts[0]))
        (dst_path / "longest_sequence.txt").write_text(str(self.longest_sequence))
        (dst_path / "int_to_name_class_mapping.tsv").write_text(
            "\n".join(f"{k} {v}" for k, v in self.int_to_name_class_mapping.items())
        )
        (dst_path / "n_classes.txt").write_text(str(self.n_classes))
        (dst_path / "ontology_embedding_dim.txt").write_text(str(self.ontology_embedding_dim))

    @staticmethod
    def load(src_path: Path) -> "MetaData":
        """Deserialize the MetaData from the information contained inside the zipped checkpoint in src_path.

        Args:
            src_path: the root folder of the metadata inside the zipped checkpoint

        Returns:
            an instance of MetaData containing the information in the checkpoint
        """
        pylogger.debug(f"Loading MetaData from '{src_path}'")

        target_dict = ast.literal_eval((src_path / "target_dict.tsv").read_text(encoding="utf-8"))

        longest_sequence = int((src_path / "longest_sequence.txt").read_text(encoding="utf-8"))

        int_to_name_class_mapping = dict()
        lines = (src_path / "int_to_name_class_mapping.tsv").read_text(encoding="utf-8").splitlines()
        for line in lines:
            int_to_name_class_mapping[int(line.split()[0])] = line.split()[1]

        n_classes = int((src_path / "n_classes.txt").read_text(encoding="utf-8"))
        ontology_embedding_dim = int((src_path / "ontology_embedding_dim.txt").read_text(encoding="utf-8"))
        return MetaData(
            target_dicts=[target_dict],
            longest_sequence=longest_sequence,
            int_to_name_class_mapping=int_to_name_class_mapping,
            n_classes=n_classes,
            ontology_embedding_dim=ontology_embedding_dim,
        )

    def __repr__(self) -> str:
        attributes = ",\n    ".join([f"{key}={value}" for key, value in self.__dict__.items()])
        return f"{self.__class__.__name__}(\n    {attributes}\n)"


class AeonDataModule(JointDataModule):
    def __init__(self, dataset: DictConfig, num_workers: DictConfig, batch_size: DictConfig, accelerator: str):
        super().__init__(dataset, num_workers, batch_size, accelerator)

    @cached_property
    def metadata(self) -> MetaData:
        if self.datasets["train"] is None:
            self.setup(stage="fit")
        print(list(self.datasets.keys()))
        self.datasets["train"].tasks = add_class_weights_multiclass(
            self.datasets["train"], self.datasets["train"].int_to_name_class_mapping
        )
        longest_sequence = self.datasets["train"].longest_sequence
        return MetaData(
            target_dicts=self.datasets["train"].tasks,
            longest_sequence=longest_sequence,
            int_to_name_class_mapping=self.datasets["train"].int_to_name_class_mapping,
            n_classes=len(self.datasets["train"].target_mapping),
            ontology_embedding_dim=len(list(self.datasets["train"].histologic_emb_dict.values())[0]),
        )


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
    print(m.datasets)
    m.setup()
    print(m.metadata)

    m.metadata.save(scratch_dir)
    m.metadata = m.metadata.load(scratch_dir)

    for batch in tqdm(m.train_dataloader()):
        # print(batch["image_ids"])
        print(batch["tile_tensor"].shape)
        print(batch["target"])
        print(batch["site"])
        print(batch["oncotree_code"])


if __name__ == "__main__":
    main()
