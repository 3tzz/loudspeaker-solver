from pathlib import Path

import torch
import torch.nn as nn

from boomspeaver.tools.data import get_repo_dir


class Model:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device = torch.device("cpu"),
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.device = device

    def train_step(self, batch_inputs, batch_targets, features):
        self.model.train()
        batch_inputs, batch_targets, features = (
            batch_inputs.to(self.device),
            batch_targets.to(self.device),
            features.to(self.device),
        )

        assert not torch.isnan(batch_inputs).any(), "Input contains NaN"
        assert not torch.isinf(batch_inputs).any(), "Input contains Inf"
        assert not torch.isnan(batch_targets).any(), "Target contains NaN"
        assert not torch.isinf(batch_targets).any(), "Target contains Inf"

        preds = self.model(batch_inputs, time=features)
        loss = self.loss_fn(preds, batch_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def eval_step(self, batch_inputs, batch_targets, features):
        self.model.eval()
        with torch.no_grad():
            batch_inputs, batch_targets, features = (
                batch_inputs.to(self.device),
                batch_targets.to(self.device),
                features.to(self.device),
            )
            # c_value = 1.0
            #
            # batch_size = batch_inputs.shape[0]
            # z = torch.full(
            #     (batch_size, 1), c_value, dtype=torch.float32, device=self.device
            # )
            preds = self.model(batch_inputs, time=features)
            loss = self.loss_fn(preds, batch_targets)

            if (
                torch.isnan(batch_inputs).any()
                or torch.isinf(batch_inputs).any()
                or torch.isnan(loss).any()
                or torch.isinf(loss).any()
            ):
                print(batch_inputs)
                print(loss)

        return loss.item()

    def train(
        self,
        train_loader,
        cfg,
        val_loader=None,
        start_epoch=0,
        best_val_loss=float("inf"),
    ):
        checkpoint_path = cfg.output.path
        if cfg.output.repo_relative:
            checkpoint_path = Path(get_repo_dir(run_type="python")) / cfg.output.path
            checkpoint_path.mkdir(parents=True, exist_ok=True)
        for epoch in range(start_epoch, cfg.epochs):
            train_losses = []
            total_samples = 0
            for batch in train_loader:
                inputs, targets, features = batch
                if inputs.ndim == 2:
                    inputs = inputs.unsqueeze(1).unsqueeze(1)
                elif inputs.ndim == 3:
                    inputs = inputs.unsqueeze(2)

                if targets.ndim == 2:
                    targets = targets.unsqueeze(1).unsqueeze(1)
                elif targets.ndim == 3:
                    targets = targets.unsqueeze(2)
                batch_size = inputs.size(0)
                loss = self.train_step(inputs, targets, features)
                train_losses.append(loss)
                total_samples += batch_size

            avg_train_loss = sum(train_losses) / total_samples
            print(
                f"Epoch {epoch+1} - Train Loss: {sum(train_losses)}, {avg_train_loss:.4f}"
            )

            if val_loader:
                val_losses = []
                total_samples = 0
                for batch in val_loader:
                    inputs, targets, features = batch
                    if inputs.ndim == 2:
                        inputs = inputs.unsqueeze(1).unsqueeze(1)
                    elif inputs.ndim == 3:
                        inputs = inputs.unsqueeze(2)

                    if targets.ndim == 2:
                        targets = targets.unsqueeze(1).unsqueeze(1)
                    elif targets.ndim == 3:
                        targets = targets.unsqueeze(2)
                    batch_size = inputs.size(0)
                    loss = self.eval_step(inputs, targets, features)
                    val_losses.append(loss)
                    total_samples += batch_size

                avg_val_loss = sum(val_losses) / total_samples
                print(
                    f"Epoch {epoch+1} - Val Loss: {sum(val_losses)}, {avg_val_loss:.4f}"
                )

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    output_path = checkpoint_path / "best_model.pt"
                    self.save_checkpoint(
                        path=output_path,
                        optimizer=True,
                        epoch=epoch,
                        val_loss=best_val_loss,
                    )
                    print("✓ New best model saved.")

            if self.scheduler:
                self.scheduler.step(avg_train_loss)

    def evaluate(self, data_loader):
        self.model.eval()
        all_losses = []
        for batch in data_loader:
            inputs, targets, time_gap = batch
            loss = self.eval_step(inputs, targets)
            all_losses.append(loss)
        return sum(all_losses) / len(all_losses)

    def save_checkpoint(
        self,
        path: Path,
        optimizer: bool = True,
        epoch=None,
        val_loss=None,
    ):
        ckpt = {
            "model_state": self.model.state_dict(),
        }
        if optimizer:
            ckpt["optimizer_state"] = self.optimizer.state_dict()
        if self.scheduler:
            ckpt["scheduler_state"] = self.scheduler.state_dict()
        if epoch is not None:
            ckpt["epoch"] = epoch
        if val_loss is not None:
            ckpt["val_loss"] = val_loss

        torch.save(ckpt, str(path))

    def load_checkpoint(self, path: Path):
        assert isinstance(path, Path)
        assert path.exists()

        ckpt = torch.load(path, weights_only=True, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        # if self.optimizer and "optimizer_state" in ckpt:
        #     self.optimizer.load_state_dict(ckpt["optimizer_state"])
        #     print("✓ Optimizer state loaded.")
        #
        # if self.scheduler and "scheduler_state" in ckpt:
        #     self.scheduler.load_state_dict(ckpt["scheduler_state"])
        #     print("✓ Scheduler state loaded.")

        print(f"✓ Checkpoint loaded: {path}")
        return ckpt.get("epoch", None), ckpt.get("val_loss", None)
