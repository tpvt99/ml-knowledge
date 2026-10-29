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

class RMSprop:
    def __init__(self, 
                 params: Iterable[t.nn.parameter.Parameter],
                 lr: float = 0.01,
                 alpha: float = 0.99,
                 eps: float = 1e-08,
                 weight_decay: float = 0.0,
                 momentum: float = 0.0):
        """
        Implements RMSprop 
        """
        self.params = list(params)
        self.lr = lr
        self.eps = eps
        self.mu = momentum
        self.lmda = weight_decay
        self.alpha = alpha

        self.b = [t.zeros_like(p) for p in self.params]
        self.v = [t.zeros_like(p) for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @t.inference_mode()
    def step(self) -> None:
        for param, b, v in zip(self.params, self.b, self.v):
            grad = param.grad # make a copy
            if self.lmda != 0:
                grad = grad + self.lmda * param # a new grad, not modified

            v.copy_(self.alpha * v + (1 - self.alpha) * grad**2)
            grad = grad / (t.sqrt(v) + self.eps) # not in place

            if self.mu != 0:
                b.copy_(self.mu * b + grad)
                grad = b

            param.sub_(self.lr * grad) # in place to replace param

    def __repr__(self) -> str:
        return (
            f"RMSprop(lr={self.lr}, eps={self.eps}, momentum={self.mu}, weight_decay={self.lmda}, alpha={self.alpha})"
        )

tests.test_rmsprop(RMSprop)
tests.test_step_doesnt_modify_grads(RMSprop, momentum=0.5)

class Adam:
    def __init__(
            self,
            params: Iterable[t.nn.parameter.Parameter],
            lr: float = 0.001,
            betas: tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.0
    ):
        """
        Implements Adam 
        """

        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.lmda = weight_decay
        self.t = 1

        self.m = [t.zeros_like(param) for param in self.params]
        self.v = [t.zeros_like(param) for param in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @t.inference_mode()
    def step(self) -> None:
        for param, m, v in zip(self.params, self.m, self.v):
            grad = param.grad
            if self.lmda != 0:
                grad = grad + self.lmda * param

            m.copy_(self.beta1 * m + (1 - self.beta1) * grad)
            v.copy_(self.beta2 * v + (1 - self.beta2) * grad**2) 

            mhat = m / (1 - self.beta1**self.t)
            vhat = v / (1 - self.beta2**self.t)

            param -= self.lr * mhat / (t.sqrt(vhat) + self.eps)

        self.t+=1

    def __repr__(self) -> str:
        return f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}, weight_decay={self.lmda})"
    
tests.test_adam(Adam)
tests.test_step_doesnt_modify_grads(Adam)

class AdamW:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
    ):
        """Implements AdamW.

        Like the PyTorch version, but assumes amsgrad=False and maximize=False
            https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
        """
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.lmda = weight_decay
        self.t = 1

        self.m = [t.zeros_like(p) for p in self.params]
        self.v = [t.zeros_like(p) for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @t.inference_mode()
    def step(self) -> None:
        for param, m, v in zip(self.params, self.m, self.v):
            grad = param.grad

            m.copy_(self.beta1 * m + (1 - self.beta1) * grad)
            v.copy_(self.beta2 * v + (1-self.beta2)*grad.pow(2))

            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t) 

            param -= self.lr * (m_hat / (t.sqrt(v_hat) + self.eps) + self.lmda * param)

        self.t += 1

    def __repr__(self) -> str:
        return f"AdamW(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}, weight_decay={self.lmda})"


tests.test_adamw(AdamW)
tests.test_step_doesnt_modify_grads(AdamW)


# %%
def opt_fn(
    fn: Callable,
    xy: Tensor,
    optimizer_class,
    optimizer_hyperparams: dict,
    n_iters: int = 100,
) -> Tensor:
    """Optimize a given function starting from the specified point.

    optimizer_class: one of the optimizers you've defined, either SGD, RMSprop, Adam or AdamW
    optimizer_hyperparams: keyword arguments passed to your optimiser (e.g. lr and weight_decay)
    """
    assert xy.requires_grad

    optimizer = optimizer_class([xy], **optimizer_hyperparams)

    xy_list = [xy.detach().clone()]  # so that we don't unintentionally modify past values in `xy_list`

    for i in range(n_iters):
        fn(xy[0], xy[1]).backward()
        optimizer.step()
        optimizer.zero_grad()
        xy_list.append(xy.detach().clone())

    return t.stack(xy_list)


points = []

optimizer_list = [
    (SGD, {"lr": 0.03, "momentum": 0.99}),
    (RMSprop, {"lr": 0.05, "alpha": 0.99, "momentum": 0.8}),
    (Adam, {"lr": 0.2, "betas": (0.99, 0.99), "weight_decay": 0.05}),
    (AdamW, {"lr": 0.3, "betas": (0.99, 0.99), "weight_decay": 0.05}),
]

for optimizer_class, params in optimizer_list:
    xy = t.tensor([2.5, 2.5], requires_grad=True)
    xys = opt_fn(
        pathological_curve_loss,
        xy=xy,
        optimizer_class=optimizer_class,
        optimizer_hyperparams=params,
    )
    points.append((xys, optimizer_class, params))

plot_fn_with_points(pathological_curve_loss, min_points=[(0, "y_min")], points=points)
#plot_fn_with_points(pathological_curve_loss, min_points=[(0, "y_min")], points=points)

# %%
def bivariate_gaussian(x, y, x_mean=0.0, y_mean=0.0, x_sig=1.0, y_sig=1.0):
    norm = 1 / (2 * np.pi * x_sig * y_sig)
    x_exp = 0.5 * ((x - x_mean) ** 2) / (x_sig**2)
    y_exp = 0.5 * ((y - y_mean) ** 2) / (y_sig**2)
    return norm * t.exp(-x_exp - y_exp)


means = [(1.0, -0.5), (-1.0, 0.5), (-0.5, -0.8)]

def neg_trimodal_func(x, y):
    """
    This function has 3 global minima, at `means`. Unstable methods can overshoot these minima, and
    non-adaptive methods can fail to converge to them in the first place given how shallow the
    gradients are everywhere except in the close vicinity of the minima.
    """
    z = -bivariate_gaussian(x, y, x_mean=means[0][0], y_mean=means[0][1], x_sig=0.2, y_sig=0.2)
    z -= bivariate_gaussian(x, y, x_mean=means[1][0], y_mean=means[1][1], x_sig=0.2, y_sig=0.2)
    z -= bivariate_gaussian(x, y, x_mean=means[2][0], y_mean=means[2][1], x_sig=0.2, y_sig=0.2)
    return z


plot_fn(neg_trimodal_func, x_range=(-2, 2), y_range=(-2, 2), min_points=means)
# %%
for optimizer_class, params in optimizer_list:
    xy = t.tensor([2.5, 2.5], requires_grad=True)
    xys = opt_fn(
        neg_trimodal_func,
        xy=xy,
        optimizer_class=optimizer_class,
        optimizer_hyperparams=params,
    )
    points.append((xys, optimizer_class, params))

plot_fn_with_points(neg_trimodal_func, min_points=means, points=points)
# %%
temp = [t.tensor([1.20, 2.0])]
if isinstance(temp, (list, tuple)):
    params = [{"params": temp}]

class hehe(t.nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = t.nn.Parameter(t.rand(10))

    def forward(self, x):
        return self.weight *x

temp = hehe()
parameter = temp.parameters()
params = None
if not isinstance(parameter, (list, tuple)):
    params = [{"params": parameter}]
print(isinstance(temp, Iterable))
print(not isinstance(temp, Iterable))
print(not isinstance(temp, (list, tuple)))
print(params)
# %%
