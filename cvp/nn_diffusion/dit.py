# Adapted from CleanDiffuser (https://github.com/CleanDiffuserTeam/CleanDiffuser), Apache-2.0; see THIRD_PARTY_LICENSES.md.
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from cvp.nn_diffusion.base_nn_diffusion import BaseNNDiffusion
from cvp.utils import SinusoidalEmbedding


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """ A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning. """

    def __init__(self, hidden_size: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, n_heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        def approx_gelu(): return nn.GELU(approximate="tanh")

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4), approx_gelu(), nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=1)
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(x, x, x)[0]
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class PatchEmbed3D(nn.Module):
    """Video to Patch Embedding.

    Args:
        patch_size (tuple): Patch token size (T, H, W). Default: (1, 2, 2).
        in_chans (int): Number of input video channels. Default: 16.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self,
        patch_size=(1, 2, 2),  # (T, H, W)
        in_chans=16,
        embed_dim=96,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # (B C T H W)
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCTHW -> BNC
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(BaseNNDiffusion):
    """DiT over a clip of image latents, with spatial and temporal sinusoidal position embeddings."""

    def __init__(
        self,
        in_dim: int,
        emb_dim: int,
        d_model: int = 384,
        n_heads: int = 6,
        depth: int = 12,
        input_size: tuple = (16, 8, 8),  # (T, H, W)
        patch_size: tuple = (1, 2, 2),  # (T, H, W)
        dropout: float = 0.1,
        timestep_emb_type: str = "positional",
        timestep_emb_params: Optional[dict] = None
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_dim, self.emb_dim = in_dim, emb_dim
        self.d_model = d_model

        self.x_embedder = PatchEmbed3D(patch_size=patch_size, in_chans=in_dim, embed_dim=d_model)
        self.num_patches = np.prod([input_size[i] // patch_size[i] for i in range(3)])
        self.num_temporal = input_size[0] // patch_size[0]
        self.num_spatial = self.num_patches // self.num_temporal

        self.map_emb = nn.Sequential(
            nn.Linear(emb_dim, d_model), nn.Mish(), nn.Linear(d_model, d_model), nn.Mish())

        self.pos_emb_temporal = SinusoidalEmbedding(d_model)
        self.pos_emb_spatial = SinusoidalEmbedding(d_model // 2)

        self.pos_embed_spatial_cache = None
        self.pos_embed_temporal_cache = None

        self.blocks = nn.ModuleList([
            DiTBlock(d_model, n_heads, dropout) for _ in range(depth)])
        self.final_layer = FinalLayer(d_model, np.prod(patch_size), in_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize time step embedding MLP:
        nn.init.normal_(self.map_emb[0].weight, std=0.02)
        nn.init.normal_(self.map_emb[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify3D(self, x: torch.Tensor):
        """
        Input:
            x: (b, t * (h/p)^2, c * p^2)
        Output:
            x: (b, t, c, h, w)
        """
        c = self.in_dim
        t, h, w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        pt, ph, pw = self.patch_size

        x = x.reshape(shape=(x.shape[0], t, h, w, pt, ph, pw, c))
        x = rearrange(x, "b t h w r p q c -> b c t h w r p q")
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
        x = rearrange(imgs, "b c t h w -> b t c h w")
        return x

    def get_pos_embed_spatial(self, x: torch.Tensor):
        h = w = torch.arange(self.input_size[1] // self.patch_size[1], device=x.device)
        grid = torch.stack(torch.meshgrid(w, h, indexing="ij"), dim=0)
        grid = grid.reshape(2, 1, self.input_size[1] // self.patch_size[1], self.input_size[2] // self.patch_size[2])

        h_ = grid[0].reshape(-1)
        emb_h = self.pos_emb_spatial(h_)
        w_ = grid[1].reshape(-1)
        emb_w = self.pos_emb_spatial(w_)
        self.pos_embed_spatial_cache = torch.cat([emb_h, emb_w], dim=1)

    def get_pos_embed_temporal(self, x: torch.Tensor):
        t = torch.arange(self.num_temporal, device=x.device)
        self.pos_embed_temporal_cache = self.pos_emb_temporal(t)

    def forward(self,
                x: torch.Tensor, noise: torch.Tensor,
                condition: Optional[torch.Tensor] = None):
        """
        Input:
            x:          (b, horizon, c, h, w)
            noise:      (b, )
            condition:  (b, emb_dim), not used (unconditional model)

        Output:
            y:          (b, horizon, c, h, w)
        """
        if self.pos_embed_spatial_cache is None:
            self.get_pos_embed_spatial(x)
        if self.pos_embed_temporal_cache is None:
            self.get_pos_embed_temporal(x)

        x = rearrange(x, 'b t c h w -> b c t h w')
        x = self.x_embedder(x)
        x = rearrange(x, "b (t s) d -> b t s d", t=self.num_temporal, s=self.num_spatial)
        x = x + self.pos_embed_spatial_cache[None,]
        x = rearrange(x, "b t s d -> b s t d")
        x = x + self.pos_embed_temporal_cache[None,]
        x = rearrange(x, "b s t d -> b (t s) d")

        emb = self.map_noise(noise)
        emb = self.map_emb(emb)
        for block in self.blocks:
            x = block(x, emb)
        x = self.final_layer(x, emb)
        x = self.unpatchify3D(x)
        return x
