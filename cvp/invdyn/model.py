import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Minimal residual block: Conv, BN, ReLU, Conv, BN plus a skip connection."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        # Use a 1x1 convolution as the skip path when the channel count changes
        self.down = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.down(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class InverseDynamicsModel(nn.Module):
    """
    Siamese residual CNN encoder followed by an MLP head.
    Works on latent states s and s_next of shape
      - [B, C, H, W]  or
      - [B, T, C, H, W] (time dimension T).
    Returns predicted actions with shape
      - [B, action_dim]      or
      - [B, T, action_dim].
    """

    def __init__(
        self,
        in_channels: int = 16,
        embed_dim: int = 128,
        action_dim: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        # Siamese residual CNN encoder
        self.encoder = nn.Sequential(
            BasicBlock(in_channels, 32),     # 16x16, 32 channels
            BasicBlock(32, 64),              # 16x16, 64 channels
            nn.AdaptiveAvgPool2d(1),         # global average pooling, [B, 64, 1, 1]
            nn.Flatten(),                    # [B, 64]
            nn.Linear(64, embed_dim, bias=False),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # MLP head on the concatenated features of s and s_next
        fuse_in = embed_dim * 2
        self.head = nn.Sequential(
            nn.Linear(fuse_in, 256, bias=False),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128, bias=False),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, action_dim)
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(
        self,
        s: torch.Tensor,
        s_next: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            s, s_next: latent tensors of shape [B, C, H, W] or [B, T, C, H, W]

        Returns:
            Predicted action tensor of shape [B, action_dim] or [B, T, action_dim]
        """
        if s.dim() == 5:  # Flatten the time dimension if present
            B, T, C, H, W = s.shape
            s = s.reshape(B * T, C, H, W)
            s_next = s_next.reshape(B * T, C, H, W)
            reshape_back = True
        else:
            reshape_back = False

        f1 = self.encoder(s)        # [N, embed_dim]
        f2 = self.encoder(s_next)   # [N, embed_dim]

        z = torch.cat([f1, f2], dim=-1)

        if reshape_back:
            z = z.view(B, T, -1)

        return self.head(z)         # [B, action_dim] or [B, T, action_dim]
