"""V2 uses the exact V1 directed attention blocks, with an optional training head."""
import torch
from torch import nn

from v1.model import GraphTransformer


class BinaryHeads(nn.Module):
    def __init__(self, main, size, seed):
        super().__init__()
        self.main = main
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 8191)
            self.auxiliary = nn.Linear(size,1)

    def forward(self, representation):
        return torch.cat([self.main(representation),self.auxiliary(representation)],dim=-1)


def build(features, cfg, pool, seed, auxiliary=False):
    model = GraphTransformer(features,cfg["width"],cfg["heads"],cfg["layers"],pool)
    if auxiliary:
        model.classifier = BinaryHeads(model.classifier,2*cfg["width"],seed)
    return model


def primary(z):
    return z if z.ndim == 1 else z[:,0]

