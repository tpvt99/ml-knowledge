#%%
import json
import sys
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path

import einops
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from IPython.display import display
from jaxtyping import Float, Int
from PIL import Image
from rich import print as rprint
from rich.table import Table
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models
from torchvision.transforms import v2 as transforms
from tqdm.auto import tqdm

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part2_cnns"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

MAIN = __name__ == "__main__"

import part2_cnns.tests as tests
import part2_cnns.utils as utils
from fundamentals_utils import CIFAR10
from plotly_utils import line

# On machines with many cores the default of one
# thread per core makes small CPU tensor operations much slower
t.set_num_threads(min(4, t.get_num_threads()))

#%%
class ReLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return t.where(x < 0, 0.0, x)

tests.test_relu(ReLU)

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        """
        A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with Pytorch.
        If `bias` is False, set `self.bias` to None 
        """
        super().__init__()
        self.weight = t.empty(out_features, in_features)
        self.weight.uniform_(-1/t.sqrt(t.tensor(in_features)), 1/t.sqrt(t.tensor(in_features)))
        self.weight = nn.Parameter(self.weight)

        if bias:
            self.bias = t.empty(out_features)
            self.bias.uniform_(-1/t.sqrt(t.tensor(in_features)), 1/t.sqrt(t.tensor(in_features)))
            self.bias = nn.Parameter(self.bias)
        else:
            self.bias = None


    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (*, in_features) 
        Return: shape (*, out_features)
        """
        out = einops.einsum(x, self.weight, "... in, out in -> ... out")
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        wstr = f"Weight shape {self.weight.shape} min {self.weight.min()}"\
                f" max: {self.weight.max()} mean {self.weight.mean()}. \n"
        if self.bias is not None:
            bstr = f"Bias shape {self.bias.shape} min {self.bias.min()}"\
                    f" max {self.bias.max()} mean {self.bias.mean()}"
            wstr += bstr
        return wstr

tests.test_linear_parameters(Linear, bias=False)
tests.test_linear_parameters(Linear, bias=True)
tests.test_linear_forward(Linear, bias=False)
tests.test_linear_forward(Linear, bias=True)


class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: Tensor) -> Tensor:
        """
        Flatten out dimensions from start_dim to end_dim, inclusive of both 
        """
        shape = input.shape

        # Get start and end dims, handling negative indexing for end dim
        start_dim = self.start_dim
        end_dim = self.end_dim if self.end_dim >= 0 else len(shape) + self.end_dim

        # Get the shapes to the left / right of flattened dims, as well as size of flattened middle
        shape_left = shape[:start_dim]
        shape_right = shape[end_dim+1:]
        shape_middle = t.prod(t.tensor(shape[start_dim: end_dim+1])).item()

        return t.reshape(input, shape_left + (shape_middle, ) + shape_right)

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["start_dim", "end_dim"]])
# %%
x = t.rand([2,3,4,5,2,6])
flat = Flatten()
y = flat(x)
print(flat)

# %%
