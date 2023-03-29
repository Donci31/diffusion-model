import torch
from torch import nn
from position_embedding import SinusoidalPositionEmbeddings
from up_block import UpBlock
from down_block import DownBlock


class Unet(nn.Module):
    def __init__(self, image_size: int, image_channels: int):
        super().__init__()
        down_channels = (image_size, image_size * 2, image_size * 4, image_size * 8)
        up_channels = down_channels[::-1]
        out_dim = 1
        time_emb_dim = 32

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        self.downs = nn.ModuleList([DownBlock(prev, lat, time_emb_dim)
                                    for prev, lat in zip(down_channels, down_channels[1:])])

        self.ups = nn.ModuleList([UpBlock(prev, lat, time_emb_dim)
                                  for prev, lat in zip(up_channels, up_channels[1:])])

        self.output = nn.Conv2d(up_channels[-1], 3, out_dim)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.conv0(x)
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)
