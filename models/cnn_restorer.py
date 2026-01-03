import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class ResNetRestoration(nn.Module):
    def __init__(self, num_blocks=10, channels=64):
        super().__init__()

        self.entry = nn.Conv2d(3, channels, 3, padding=1)

        self.resblocks = nn.Sequential(
            *[ResBlock(channels) for _ in range(num_blocks)]
        )

        self.exit = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x):
        out = self.entry(x)
        out = self.resblocks(out)
        out = self.exit(out)

        # Global residual learning
        return x + out
