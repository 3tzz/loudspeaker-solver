from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from boomspeaver.structural.surrogate.dataset import get_dataset
from boomspeaver.structural.surrogate.model import Model
from boomspeaver.structural.surrogate.models import ModelArchitecture
from boomspeaver.structural.surrogate.utils import save_config
from boomspeaver.tools.data import search4paths


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
# @hydra.main(config_path=None, config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:
    print("Loading data...")
    dataset_train, dataset_val = get_dataset(cfg)
    print(f"Data loaded: train {len(dataset_train)}, validate {len(dataset_val)}.")

    print("Dataset loading...")
    train_loader = DataLoader(
        dataset_train, batch_size=cfg.training.batch_size, shuffle=cfg.training.shuffle
    )

    val_loader = DataLoader(
        dataset_val, batch_size=cfg.training.batch_size, shuffle=False
    )
    print(f"Dataset loaded: train {len(train_loader)}, validate {len(val_loader)}.")

    print("Building model architecture...")
    model = ModelArchitecture.build_model(cfg.model)
    print("Model architecture builded.")

    if cfg.training.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.training.optimizer}")

    if cfg.training.loss == "mse":
        loss_fn = torch.nn.MSELoss(reduction="sum")
    elif cfg.training.loss == "l1":
        loss_fn = torch.nn.L1Loss()
    else:
        raise ValueError(f"Unsupported loss function: {cfg.training.loss}")

    if cfg.training.scheduler is False:
        scheduler = None
    else:
        scheduler = instantiate(cfg.training.scheduler, optimizer=optimizer)

    trainer = Model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=cfg.training.device,
    )

    epoch, best_val_loss = 0, float("inf")
    if cfg.training.resume:
        checkpoint_paths = list(Path(cfg.training.output.path).glob("*.pt"))
        if not checkpoint_paths:
            print(
                f"WARNING: There are no checkpoint in output path to load: {cfg.training.output.path}."
            )
        elif len(checkpoint_paths) != 1:
            raise ValueError(
                f"There are probably multiple checkpoint files in output path: {checkpoint_paths}."
            )
        else:
            epoch, best_val_loss = trainer.load_checkpoint(checkpoint_paths[0])
            epoch += 1
    save_config(cfg)

    print("Training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg.training,
        start_epoch=epoch,
        best_val_loss=best_val_loss,
    )


if __name__ == "__main__":
    train()
