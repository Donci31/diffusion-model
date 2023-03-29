from torch import nn
import torch.nn.functional as F


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)

        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm2 = nn.BatchNorm2d(out_ch)

        self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

    def forward(self, x, t):
        x = F.relu(self.conv1(x))
        x = self.bnorm1(x)

        time_emb = F.relu(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 2]
        x = x + time_emb

        x = F.relu(self.conv2(x))
        x = self.bnorm2(x)
        x = self.transform(x)

        return x
