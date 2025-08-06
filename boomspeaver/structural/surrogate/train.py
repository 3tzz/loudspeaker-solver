import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from boomspeaver.structural.surrogate.dataset import get_dataset
from boomspeaver.structural.surrogate.model import Model
from boomspeaver.structural.surrogate.models import ModelArchitecture
from boomspeaver.structural.surrogate.utils import save_config


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:
    print("Loading data...")
    dataset_train, dataset_val = get_dataset(cfg)
    print("Data loaded.")

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
        scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    trainer = Model(
        model=model, optimizer=optimizer, loss_fn=loss_fn, device=cfg.training.device
    )
    save_config(cfg)

    # for step in range(10000):
    #     for batch in train_loader:
    #         inputs, targets, _ = batch
    #
    #         inputs = inputs.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, T]
    #         targets = targets.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, T]
    #
    #         loss = trainer.train_step(inputs, targets)
    #     loss += loss
    #     print(f"Step {step} - Loss: {loss:.6f}")
    print("Training...")
    trainer.train(train_loader=train_loader, val_loader=val_loader, cfg=cfg.training)


if __name__ == "__main__":
    train()
