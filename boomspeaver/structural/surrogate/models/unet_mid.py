import torch
import torch.nn as nn

from boomspeaver.structural.surrogate.models.unet import (
    AttentionBlock,
    ConditionedBlock,
    ResidualBlock,
)


class ContinuousTimeEmbedding(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(1, emb_dim)
        self.act = nn.SiLU()  # smooth nonlinearity
        self.linear2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: shape (batch,) or (batch, 1), continuous time in seconds
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        emb = self.linear1(t)
        emb = self.act(emb)
        emb = self.linear2(emb)
        return emb


class MiddleBlock(ConditionedBlock):
    """Middle block It combines a `ResidualBlock`, `AttentionBlock`, followed by another
    `ResidualBlock`.

    This block is applied at the lowest resolution of the U-Net.

    Args:
        n_channels (int): Number of channels in the input and output.
        cond_channels (int): Number of channels in the conditioning vector.
        has_attn (bool, optional): Whether to use attention block. Defaults to False.
        activation (str): Activation function to use. Defaults to "gelu".
        norm (bool, optional): Whether to use normalization. Defaults to False.
        use_scale_shift_norm (bool, optional): Whether to use scale and shift approach to conditoning (also termed as `AdaGN`).
        n_dims (int): Number of spatial dimensions. Defaults to 1. Defaults to False.
    """

    def __init__(
        self,
        padding_mode: str,
        n_channels: int,
        cond_channels: int,
        has_attn: bool = True,
        activation: str = "gelu",
        norm: bool = False,
        use_scale_shift_norm: bool = False,
        n_dims: int = 1,
    ):
        super().__init__()
        self.res1 = ResidualBlock(
            padding_mode,
            n_channels,
            n_channels,
            cond_channels,
            activation=activation,
            norm=norm,
            use_scale_shift_norm=use_scale_shift_norm,
            n_dims=n_dims,
        )
        self.attn = AttentionBlock(n_channels) if has_attn else nn.Identity()
        self.res2 = ResidualBlock(
            padding_mode,
            n_channels,
            n_channels,
            cond_channels,
            activation=activation,
            norm=norm,
            use_scale_shift_norm=use_scale_shift_norm,
            n_dims=n_dims,
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        x = self.res1(x, emb)
        x = self.attn(x)
        x = self.res2(x, emb)
        return x


class UnetMid(MiddleBlock):
    def __init__(
        self,
        padding_mode,
        n_channels,
        cond_channels,
        has_attn=True,
        activation="gelu",
        norm=False,
        use_scale_shift_norm=False,
        n_dims=1,
    ):
        super().__init__(
            padding_mode,
            n_channels,
            cond_channels,
            has_attn,
            activation,
            norm,
            use_scale_shift_norm,
            n_dims,
        )
        self.time_emb = ContinuousTimeEmbedding(cond_channels)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        emb = self.time_emb(time)
        x = x.squeeze(1).squeeze(1).unsqueeze(-1)
        return super().forward(x, emb).squeeze(-1).unsqueeze(1).unsqueeze(1)

    # def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    #     emb = self.time_emb(time)
    #     x = x.squeeze(1)
    #     return super().forward(x, emb).unsqueeze(1)
