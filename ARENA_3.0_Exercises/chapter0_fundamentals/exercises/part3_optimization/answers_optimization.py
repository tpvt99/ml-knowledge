#%%
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal

import numpy as np
import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from IPython.core.display import HTML
from IPython.display import display
from jaxtyping import Float, Int
from torch import Tensor, optim
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part3_optimization"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))


MAIN = __name__ == "__main__"

import part3_optimization.tests as tests
from fundamentals_utils import IMAGENET_TRANSFORM, CIFAR10
from part2_cnns.solutions import Linear, ResNet34, get_resnet_for_feature_extraction
from part3_optimization.utils import plot_fn, plot_fn_with_points
from plotly_utils import bar, imshow, line

# Cap the number of CPU threads PyTorch uses. On machines with many cores the default of one
# thread per core makes small CPU tensor operations much slower
t.set_num_threads(min(4, t.get_num_threads()))

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# %%
def pathological_curve_loss(x: Tensor, y: Tensor):
    # Example of a pathological curvature. There are many more possible, feel free to experiment here!
    x_loss = t.tanh(x) ** 2 + 0.01 * t.abs(x)
    y_loss = t.sigmoid(y)
    return x_loss + y_loss


plot_fn(pathological_curve_loss, min_points=[(0, "y_min")])

# %%
def opt_fn_with_sgd(
        fn: Callable, xy: Float[Tensor, "2"], lr=0.001, momentum=0.98, n_iters: int = 1000
) -> Float[Tensor, "n_iters_plus_1 2"]:
    """
    Optimize a given function starting from the specified point.

    xy: shape (2,). The (x, y) starting point
    n_iters: number of steps
    lr, momentum: parameters passed to the torch.optim.SGD optimizer

    Return (n_iters+1, 2). The (x,y) values, from initial values to values after step `n_iters`
    """
    # Make sure tensor has requires_grad=True, otherwise it can't be optimized
    xy.requires_grad_(True).to(device)
    optimizer = optim.SGD(params=[xy], lr=lr, momentum=momentum)

    points = [xy.cpu().detach().clone()]
    for iter in range(n_iters):
        loss = pathological_curve_loss(xy[0], xy[1])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        points.append(xy.cpu().detach().clone())

    return t.stack(points, dim=0)

points = []

optimizer_list = [
    (optim.SGD, {"lr": 0.1, "momentum": 0.0}),
    (optim.SGD, {"lr": 0.02, "momentum": 0.99}),
]

for optimizer_class, params in optimizer_list:
    xy = t.tensor([2.5, 2.5], requires_grad=True)
    xys = opt_fn_with_sgd(pathological_curve_loss, xy=xy, lr=params["lr"], momentum=params["momentum"])
    points.append((xys, optimizer_class, params))
    print(f"{params=}, last point={xys[-1]}")

plot_fn_with_points(pathological_curve_loss, points=points, min_points=[(0, "y_min")])
# %%
class SGD:
    def __init__(self,
                 params: Iterable[t.nn.parameter.Parameter],
                 lr: float,
                 momentum: float = 0.0,
                 weight_decay: float = 0.0,
    ):
        """
        Implements SGD with momentum

        Like the Pytorch version, but assume nesterov=False, maximize=False, and dampening=0 
        """
        self.params = list(params) # turns params into a list (it might be a generator so iterating over it empties it)
        self.lr = lr
        self.mu = momentum
        self.lmda = weight_decay

        self.b = [t.zeros_like(p) for p in self.params]

    def zero_grad(self) -> None:
        """Zeros all gradients of the parameters in `self.params`."""
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        """Performs a single optimization step of the SGD algorithm"""
        for i, param in enumerate(self.params):
            grad = param.grad
            if self.lmda != 0:
                grad = grad + self.lmda * param # should not be in place as we dont want to change param.grad

            if abs(self.mu) > 1e-6:
                #self.b[i] = self.mu * self.b[i] + grad # it seems must use in-place but self.b is temporary can use non inplace can?
                self.b[i].copy_(self.mu * self.b[i] + grad) # oh the .copy_ actually creates new object, better use .mul_().add_()
                grad = self.b[i]

            param.add_(-self.lr * grad) 

    def __repr__(self) -> str:
        return f"SGD(lr={self.lr}, momentum={self.mu}, weight_decay={self.lmda})"

tests.test_sgd(SGD)
tests.test_step_doesnt_modify_grads(SGD, momentum=0.5)

# %%
x = t.tensor([0.2, 0.3]).requires_grad_(True)
optim = SGD([x], lr=0.01)
loss = (x - t.tensor([1.0, 1.0]))**2
loss.sum().backward()
print(x)
print(f"id x: {x.data_ptr()} id x's grad {x.grad.data_ptr()} id x'data {x.data.data_ptr()}")
print(x.grad)
print(id(x.grad))
print(id(x.data))
optim.step()
#optim.zero_grad()
print(x)
print(x.grad)
print(x.data)
print(f"id x: {x.data_ptr()} id x's grad {x.grad.data_ptr()} id x'data {x.data.data_ptr()}")

# %%
w = t.nn.Parameter(t.tensor([1.0, 2.0]), requires_grad=False)
alias = w.detach()          # shares w's storage, a separate object

w.data = w.data - 1         # swaps in new storage
print(w, alias)             # w is [0, 1], alias is still [1, 2]

w = t.nn.Parameter(t.tensor([1.0, 2.0]))
alias = w.detach()
with t.no_grad():
    w -= 1                  # overwrites the shared storage
print(w, alias)             # both are [0, 1]
# %%
