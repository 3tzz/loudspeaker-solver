import random
from pathlib import Path

import numpy as np
import omegaconf
import torch
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from boomspeaver.tools.data import (
    get_repo_dir,
    load_numpy_file,
    load_txt_file,
    save_txt_file,
    search4paths,
)


class DiaphragmDataset(Dataset):
    def __init__(
        self,
        input_data: list[tuple[torch.Tensor, torch.Tensor]],
        global_ids: list[str] | None = None,
        device: torch.device = torch.device("cpu"),
    ):

        self.device = device
        self.data = input_data
        self.global_ids = self._get_global_ids(global_ids)
        self.validate()

    def validate(self) -> None:
        """
        Validate input data
        """
        assert isinstance(self.device, torch.device)
        assert self.device.type in {"cpu", "gpu"}
        assert isinstance(self.data, list)
        assert len(self.data) != 0
        assert isinstance(self.data[0], tuple)
        assert len(self.data[0]) == 2
        assert isinstance(self.data[0][0], torch.Tensor) and isinstance(
            self.data[0][1], torch.Tensor
        )
        assert isinstance(self.global_ids, list)
        assert len(self.global_ids) != 0
        assert isinstance(self.global_ids[0], str)
        assert len(list(self.global_ids)) == len(list(set(self.global_ids)))
        assert len(self.data) == len(self.global_ids)

    def _get_global_ids(self, global_ids: list[str] | None) -> list[str]:
        if global_ids is None:
            global_ids = [str(i) for i in self.data]
        return global_ids

    @classmethod
    def from_time_history(
        cls,
        input_data: list[np.ndarray],
        global_ids: list[str] | None = None,
        time_history: int = 2,
        time_future: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> "DiaphragmDataset":
        assert isinstance(input_data, list)
        assert len(input_data) != 0
        assert isinstance(input_data[0], np.ndarray)
        assert len(input_data[0].shape) == 2
        assert isinstance(global_ids, list) or global_ids is None
        assert isinstance(time_history, int)
        assert isinstance(time_future, int)
        if isinstance(global_ids, list):
            assert len(global_ids) == len(input_data)

        data = []
        global_idss = []
        for idx, data_i in enumerate(input_data):
            for i in range(time_history, data_i.shape[0] - time_future):
                input_tensor = torch.tensor(
                    data_i[i - time_history : i], dtype=torch.float32
                )
                output_tensor = torch.tensor(
                    data_i[i : i + time_future], dtype=torch.float32
                )
                data.append((input_tensor, output_tensor))
                if global_ids:
                    global_idss.append(global_ids[idx] + f"_{str(i)}")

        if global_idss:
            return cls(input_data=data, global_ids=global_idss, device=device)
        else:
            return cls(input_data=data, device=device)

    @classmethod
    def from_list2stepbystep(
        cls,
        input_data: list[np.ndarray],
        global_ids: list[str] | None = None,
        device: torch.device = torch.device("cpu"),
    ) -> "DiaphragmDataset":
        """
        Get DiaphragmDataset from array loaded
        from simulated data "[n_simulation,[time_step, x_displacement]]".
        Changes input to data pairs step by step.
        """
        assert isinstance(input_data, list)
        assert len(input_data) != 0
        assert isinstance(input_data[0], np.ndarray)
        assert len(input_data[0].shape) == 2
        assert isinstance(global_ids, list) or global_ids is None
        if isinstance(global_ids, list):
            assert len(global_ids) == len(input_data)

        data = []
        global_idss = []
        for idx, data_i in enumerate(input_data):
            for i in range(data_i.shape[0] - 1):
                input_tensor = torch.tensor(data_i[i], dtype=torch.float32)
                output_tensor = torch.tensor(data_i[i + 1], dtype=torch.float32)
                data.append((input_tensor, output_tensor))
                if global_ids:
                    global_idss.append(global_ids[idx] + f"_{str(i)}")
        if global_idss:
            return cls(input_data=data, global_ids=global_idss, device=device)
        else:
            return cls(input_data=data, device=device)

    @classmethod
    def from_config(
        cls,
        cfg: DictConfig,
        device: torch.device = torch.device("cpu"),
    ) -> "DiaphragmDataset":
        """
        Get DiaphragmDataset from provided dataset config.
        """
        assert isinstance(cfg, DictConfig)
        assert isinstance(cfg.data, DictConfig)
        assert isinstance(cfg.dataset, DictConfig)
        assert isinstance(cfg.data.input_path, str)
        assert isinstance(cfg.data.repo_relative, bool)
        assert isinstance(cfg.data.recursive_search, bool)
        input_directory = Path(cfg.data.input_path)
        if cfg.data.repo_relative:
            input_directory = get_repo_dir(run_type="python") / input_directory
        assert isinstance(input_directory, Path)
        assert input_directory.exists()
        assert isinstance(
            cfg.data.search_patterns, (list, omegaconf.listconfig.ListConfig)
        )
        assert len(cfg.data.search_patterns) != 0
        assert isinstance(cfg.data.search_patterns[0], str)

        paths = []
        for pattern in cfg.data.search_patterns:
            paths += search4paths(
                input_path=input_directory,
                search_pattern=pattern,
                recursive=cfg.data.recursive_search,
            )
        arrays = []
        global_ids = []
        for path in paths:
            try:
                arrays.append(load_numpy_file(path))
            except:
                print(path)
            parts = []
            for part_cfg in cfg.data.global_id_formatter.parts.values():
                if part_cfg.type == "from_path":
                    part = path.parts[part_cfg.part_index].split(
                        part_cfg.split_delimiter
                    )[part_cfg.split_index]
                    parts.append(part)
            global_ids.append(cfg.data.global_id_formatter.join_delimiter.join(parts))

        if cfg.dataset.type == "time":
            return cls.from_time_history(
                input_data=arrays,
                global_ids=global_ids,
                device=device,
                time_history=cfg.dataset.time_history,
                time_future=cfg.dataset.time_future,
            )

        elif cfg.dataset.type == "statebystate":
            return cls.from_list2stepbystep(
                input_data=arrays, global_ids=global_ids, device=device
            )
        else:
            raise NotImplementedError(f"Unknown dataset type: {cfg.dataset.type}.")

    def save_global_ids(self, output_path: Path = Path("global_ids.csv")):
        """
        Save global_ids to an external CSV file
        to ensure reproducibility for the same global ids formatters.
        """
        save_txt_file(data=self.global_ids, file_path=output_path)

    def filter(self, global_ids: set[str]) -> "DiaphragmDataset":
        """
        Return a new DiaphragmDataset instance containing only samples
        whose global_ids are in the provided as argument.
        """
        assert isinstance(global_ids, set)
        filtered_data = []
        filtered_ids = []

        for i, gid in enumerate(self.global_ids):
            if gid in global_ids:
                filtered_data.append(self.data[i])
                filtered_ids.append(gid)

        return DiaphragmDataset(
            input_data=filtered_data,
            global_ids=filtered_ids,
            device=self.device,
        )

    def split(
        self, val_ratio: float = 0.2, seed: int = 42
    ) -> tuple["DiaphragmDataset", "DiaphragmDataset"]:
        """Split DiaphragmDataset for training and validation Diaphragmsdatasets."""

        train_ids, val_ids = train_test_split(
            self.global_ids, test_size=val_ratio, random_state=seed
        )
        return (
            self.filter(set(train_ids)),
            self.filter(set(val_ids)),
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_tensor, output_tensor = self.data[idx]
        global_id = self.global_ids[idx]
        return input_tensor.to(self.device), output_tensor.to(self.device), global_id


class DatasetNormalizer:
    def __init__(self, method="standard", device=torch.device("cpu")):
        assert method in [
            "standard",
            "minmax",
            "unit",
        ], "Method must be 'standard', 'minmax' or 'unit'"
        self.method = method
        self.params = {}
        self.device = device

    def fit(self, dataset: DiaphragmDataset):
        """
        Compute normalization parameters from all input tensors of the dataset.
        """
        inputs = torch.stack([x[0] for x in dataset.data]).to(self.device)

        if self.method == "standard":
            self.params["mean"] = inputs.mean(dim=0)
            self.params["std"] = inputs.std(dim=0)
        elif self.method == "minmax":
            self.params["min"] = inputs.min(dim=0).values
            self.params["max"] = inputs.max(dim=0).values
        elif self.method == "unit":
            self.params["norms"] = torch.norm(inputs, dim=1)

    def transform(self, dataset: DiaphragmDataset, inplace=True):
        """
        Apply normalization to the dataset inputs (and optionally outputs).
        If inplace=True, modifies dataset.data in place.
        """
        eps = 1e-8

        def normalize_tensor(t):
            if self.method == "standard":
                return (t - self.params["mean"]) / (self.params["std"] + eps)
            elif self.method == "minmax":
                return (t - self.params["min"]) / (
                    self.params["max"] - self.params["min"] + eps
                )
            elif self.method == "unit":
                norm = torch.norm(t) + eps
                return t / norm

        if inplace:
            new_data = []
            for input_t, output_t in dataset.data:
                norm_input = normalize_tensor(input_t.to(self.device))
                # Optionally normalize output_t too; here left as is:
                new_data.append((norm_input.cpu(), output_t))
            dataset.data = new_data
        else:
            norm_data = []
            for input_t, output_t in dataset.data:
                norm_input = normalize_tensor(input_t.to(self.device))
                norm_data.append((norm_input.cpu(), output_t))
            return norm_data

    def fit_transform(self, dataset: DiaphragmDataset, inplace=True):
        self.fit(dataset)
        self.transform(dataset, inplace=inplace)


class DatasetNormalizerPerPair:
    def __init__(self, to_val: float = 1.0, device=torch.device("cpu")):
        self.device = device
        self.to_val = to_val
        self.params = {}

    def fit_transform(self, dataset: DiaphragmDataset, inplace=True):
        """
        Apply per-sample min-max normalization to inputs and outputs.
        Stores min/max per sample for possible denormalization.
        """
        norm_data = []
        stats = []

        for input_t, output_t in dataset.data:
            input_t = input_t.to(self.device)
            output_t = output_t.to(self.device)

            concatenated = torch.cat([input_t, output_t], dim=0)

            in_min = concatenated.min()
            in_max = concatenated.max()

            norm_input = (input_t - in_min) * self.to_val / (in_max - in_min + 1e-8)
            norm_output = (output_t - in_min) * self.to_val / (in_max - in_min + 1e-8)

            norm_data.append((norm_input.cpu(), norm_output.cpu()))
            stats.append((in_min, in_max))

        if inplace:
            dataset.data = norm_data
        else:
            return norm_data


def get_dataset(cfg: DictConfig) -> tuple["DiaphragmDataset", "DiaphragmDataset"]:
    dataset_train = DiaphragmDataset.from_config(cfg)
    if cfg.dataset.train_val.split is True:
        if cfg.dataset.train_val.external is False:
            dataset_train, dataset_val = dataset_train.split(
                val_ratio=cfg.dataset.train_val.ratio, seed=cfg.dataset.train_val.seed
            )
        else:

            val_path = Path(cfg.dataset.train_val.input_val.path)
            if cfg.dataset.train_val.input_val.repo_relative:
                val_path = Path(get_repo_dir(run_type="python")) / val_path

            global_ids_val = set(load_txt_file(val_path))
            dataset_val = dataset_train.filter(global_ids_val)

            train_path = Path(cfg.dataset.train_val.input_train.path)
            if cfg.dataset.train_val.input_train.repo_relative:
                train_path = Path(get_repo_dir(run_type="python")) / train_path

            global_ids_train = set(load_txt_file(train_path))
            dataset_train = dataset_train.filter(global_ids_train)
    else:
        dataset_val = dataset_train.filter(
            set(random.sample(dataset_train.global_ids, 64))
        )

    if cfg.dataset.global_ids.save is True:
        output_path = Path(cfg.dataset.global_ids.output.path, "global_ids_train.txt")
        if cfg.dataset.global_ids.output.repo_relative:
            output_path = Path(get_repo_dir(run_type="python"), output_path)
        if cfg.dataset.global_ids.output.exists_ok is True and output_path.exists():
            raise ValueError(
                f"File with saved global ids already exists: {str(output_path)}. Provide different output path or remove unnecessary file."
            )

        dataset_train.save_global_ids(output_path=output_path)
        output_path = output_path.parent / "global_ids_val.txt"
        if cfg.dataset.global_ids.output.exists_ok is True and output_path.exists():
            raise ValueError(
                f"File with saved global ids already exists: {str(output_path)}. Provide different output path or remove unnecessary file."
            )
        dataset_val.save_global_ids(output_path=output_path)

    if cfg.dataset.normalization:
        if cfg.dataset.normalization == "perpair":
            normalizer = DatasetNormalizerPerPair(
                to_val=10.0, device=torch.device("cpu")
            )
            normalizer.fit_transform(dataset_val)
            normalizer.fit_transform(dataset_train)
        else:
            normalizer = DatasetNormalizer(
                method=cfg.dataset.normalization, device=torch.device("cpu")
            )
            normalizer.fit_transform(dataset_val)
            normalizer.fit_transform(dataset_train)
    return dataset_train, dataset_val
