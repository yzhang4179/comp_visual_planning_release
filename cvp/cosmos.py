import einops
import numpy as np
import torch


class CICosmos:
    """
    Continuous image tokenizer `Cosmos-Tokenizer-CI16x16` of NVIDIA Cosmos, released under the NVIDIA Open Model License.
    A 256x256 image becomes a (16, 16, 16) latent. The TorchScript encoder and decoder run in bfloat16.
    """

    def __init__(self, encoder_path: str, decoder_path: str, device='cpu'):
        self.encoder = torch.jit.load(encoder_path, map_location=device).eval().to(torch.bfloat16)
        self.decoder = torch.jit.load(decoder_path, map_location=device).eval().to(torch.bfloat16)
        self.device = device
        self.scale = 0.5  # latents are multiplied by this before the diffusion model sees them

    @torch.no_grad()
    def encode(self, img):
        assert len(img.shape) == 4 or len(img.shape) == 5, f'Input shape should be either (B, C, H, W) or (B, T, C, H, W), but got {img.shape}'
        img_dim = img.dim()
        b = img.shape[0]
        if img_dim == 5:
            img = einops.rearrange(img, 'b t c h w -> (b t) c h w')
        latent = self.encoder(img)[0]
        if img_dim == 5:
            latent = einops.rearrange(latent, '(b t) c h w -> b t c h w', b=b)
        return latent * self.scale

    @torch.no_grad()
    def decode(self, latent):
        assert len(latent.shape) == 4 or len(latent.shape) == 5, f'Input shape should be either (B, c, h, w) or (B, t, c, h, w), but got {latent.shape}'
        latent = latent / self.scale
        latent_dim = latent.dim()
        b = latent.shape[0]
        if latent_dim == 5:
            latent = einops.rearrange(latent, 'b t c h w -> (b t) c h w')
        img = self.decoder(latent)
        if latent_dim == 5:
            img = einops.rearrange(img, '(b t) c h w -> b t c h w', b=b)
        return torch.clamp(img, -1, 1)


def video2latent(x: np.ndarray, model: CICosmos) -> torch.Tensor:
    '''
    x: (H, W, 3) or (T, H, W, 3) uint8 images
    return: (1, C, h, w) or (1, T, C, h, w) float32 latents
    '''
    assert x.ndim in [3, 4], f'Input array must be 3D or 4D, but got {x.ndim}D'
    # [0, 255] -> [-1, 1]
    x = torch.from_numpy(x.astype(np.float32)).to(model.device)
    x = x / 255 * 2 - 1
    if x.dim() == 3:
        x = x[None, ...].permute(0, 3, 1, 2)
    elif x.dim() == 4:
        x = x[None, ...].permute(0, 1, 4, 2, 3)
    x = model.encode(x.to(torch.bfloat16)).to(torch.float32)
    return x


def latent2video(model: CICosmos, x: torch.Tensor) -> np.ndarray:
    '''
    x: (1, T, C, h, w) latents
    return: (T, H, W, 3) uint8 images
    '''
    x = model.decode(x.to(model.device).to(torch.bfloat16))
    x = x.permute(0, 1, 3, 4, 2)  # (b, t, H, W, 3)
    x = x.to(torch.float32)
    x = torch.clamp(x, -1, 1)
    x = x.squeeze().cpu().detach().numpy()
    # [-1, 1] -> [0, 255]
    x = (x + 1) / 2 * 255
    x = x.astype(np.uint8)
    return x
