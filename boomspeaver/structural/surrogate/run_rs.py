import random
import shutil
from pathlib import Path

import hydra
from omegaconf import DictConfig

from boomspeaver.structural.surrogate.train import train
from boomspeaver.tools.data import get_repo_dir, load_txt_file, save_txt_file


def rs(input_files: list[str], size: int, output_file: Path):
    random_indices = random.sample(range(len(input_files)), size)
    new_set = []
    for i in random_indices:
        new_set.append(input_files[i])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    # output_file.parent.mkdir(parents=True, exist_ok=False)
    save_txt_file(new_set, output_file)
    return new_set


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def rs_training_loop(cfg: DictConfig) -> None:
    repo_dir = get_repo_dir(run_type="python")
    print(cfg)

    output_datasets = Path(cfg.al.output.path) / "datasets"
    if cfg.al.output.repo_relative:
        output_datasets = repo_dir / output_datasets
    output_datasets.mkdir(parents=True, exist_ok=True)
    # output_datasets.mkdir(parents=True, exist_ok=False)
    output_directory = output_datasets.parent

    input_train_set = Path(cfg.al.input.train_set.path)
    if cfg.al.input.train_set.repo_relative:
        input_train_set = repo_dir / input_train_set

    print(input_train_set)
    input_file_list = load_txt_file(input_train_set)

    cfg.training.device = "cuda"
    cfg.dataset.global_ids.save = False
    cfg.dataset.train_val.input_val.path = cfg.al.input.val_set.path
    cfg.dataset.train_val.input_val.repo_relative = cfg.al.input.val_set.repo_relative
    cfg.dataset.train_val.input_train.repo_relative = cfg.al.output.repo_relative
    cfg.training.epochs = 0

    checkpoint_path = ""
    # for i in range(30, cfg.al.params.iterations):
    for i in range(cfg.al.params.iterations):

        cfg.training.epochs += cfg.al.params.epoch_step
        output_dir = output_directory / f"al_{i}"
        print(f"Active learning step: {i} in output path: {output_dir}")

        output_path = output_dir / f"datasets/train_{i}.txt"
        output_dir.mkdir(parents=True, exist_ok=True)
        # output_dir.mkdir(parents=True, exist_ok=False)

        if checkpoint_path:
            shutil.copy(checkpoint_path, output_dir)
            cfg.training.resume = True

        cfg.dataset.train_val.input_train.path = output_path
        cfg.training.output.path = output_dir
        cfg.training.output.repo_relative = False
        global_ids_train = rs(input_file_list, cfg.al.params.train_size, output_path)
        global_ids_val = load_txt_file(Path(cfg.al.input.val_set.path))

        output_path = output_dir / f"datasets/whole_{i}.txt"
        save_txt_file(global_ids_val + global_ids_train, output_path)
        cfg.data.filter.input.path = output_path
        cfg.data.filter.external = True

        train(cfg)

        checkpoint_path = output_dir / "best_model.pt"


if __name__ == "__main__":
    rs_training_loop()
