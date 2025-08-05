from pathlib import Path

from omegaconf import OmegaConf

from boomspeaver.tools.data import get_repo_dir


def save_config(cfg) -> None:
    output_path = Path(cfg.training.output.path) / "config.yaml"
    if cfg.training.output.repo_relative:
        output_path = get_repo_dir(run_type="python") / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, str(output_path))
