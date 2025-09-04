from omegaconf import OmegaConf

from .nn import NN
from .unet import Unet1D
from .unet_mid import UnetMid

__all__ = ["ModelArchitecture", "NN"]

MODEL_REGISTRY = {"nn": NN, "unet": Unet1D, "unet_mid": UnetMid}


class ModelArchitecture:
    @staticmethod
    def build_model(cfg):
        model_type = cfg.model_type.lower()
        if model_type in MODEL_REGISTRY:
            kwargs = OmegaConf.to_container(cfg, resolve=True)
            kwargs.pop("model_type", None)
            return MODEL_REGISTRY[model_type](**kwargs)
        else:
            raise NotImplementedError(f"Unknown model type: {model_type}.")
