#%%
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import pandas as pd
import torch as t
from torch import Tensor, optim
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from typing import Callable, Iterable, Tuple, Optional, Type
from jaxtyping import Float
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from dataclasses import dataclass
from tqdm.notebook import tqdm
from pathlib import Path
import numpy as np
from IPython.display import display, HTML

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part4_optimization"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))
os.chdir(section_dir)

from plotly_utils import bar, imshow, plot_train_loss_and_test_accuracy_from_trainer
from part3_resnets.solutions import IMAGENET_TRANSFORM, ResNet34, get_resnet_for_feature_extraction
from part4_optimization.utils import plot_fn, plot_fn_with_points
import part4_optimization.tests as tests

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"
# %%
def pathological_curve_loss(x: t.Tensor, y: t.Tensor):
    # Example of a pathological curvature. There are many more possible, feel free to experiment here!
    x_loss = t.tanh(x) ** 2 + 0.01 * t.abs(x)
    y_loss = t.sigmoid(y)
    return x_loss + y_loss


if MAIN:
    plot_fn(pathological_curve_loss)
# %%
def opt_fn_with_sgd(fn: Callable, xy: t.Tensor, lr=0.001, momentum=0.98, n_iters: int = 100):
    '''
    Optimize the a given function starting from the specified point.

    xy: shape (2,). The (x, y) starting point.
    n_iters: number of steps.
    lr, momentum: parameters passed to the torch.optim.SGD optimizer.

    Return: (n_iters, 2). The (x,y) BEFORE each step. So out[0] is the starting point.
    '''
    outputs = []
    parameters = [t.nn.Parameter(xy[0]), t.nn.Parameter(xy[1])]
    optimizers = t.optim.SGD(params=parameters, lr=lr, momentum=momentum)


    for _ in range(n_iters):
        outputs.append(t.Tensor([i.detach() for i in parameters]))

        optimizers.zero_grad()

        loss = fn(*parameters)
        loss.backward()

        optimizers.step()

    outputs = t.stack(outputs, dim=0)
    return outputs

opt_fn_with_sgd(pathological_curve_loss, t.Tensor([2.5,2.5]), 0.02, 0.99)

#%%
if MAIN:
    points = []

    optimizer_list = [
        (optim.SGD, {"lr": 0.1, "momentum": 0.0}),
        (optim.SGD, {"lr": 0.02, "momentum": 0.99}),
    ]

    for optimizer_class, params in optimizer_list:
        xy = t.tensor([2.5, 2.5], requires_grad=True)
        xys = opt_fn_with_sgd(pathological_curve_loss, xy=xy, lr=params['lr'], momentum=params['momentum'])

        points.append((xys, optimizer_class, params))

    plot_fn_with_points(pathological_curve_loss, points=points)
# %%
class SGD:
    def __init__(
            self,
            params: Iterable[t.nn.parameter.Parameter],
            lr: float,
            momentum: float = 0.0,
            weight_decay: float = 0.0
    ):
        '''Implements SGD with momentum.

        Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
            https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD

        '''
        params = list(params) # turn params into a list (because it may be a generator)
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = 0
        self.t = 0
        self.nesterov = False

        self.bt = []
        for param in self.params:
            self.bt.append(t.zeros_like(param.data))

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None # on same device and reset to zero
            #param.grad = t.zeros_like(param.data) # on same device and reset to zero
            #param.grad_(None)
            pass

    @t.inference_mode()
    def step(self) -> None:
        self.t += 1

        if self.weight_decay != 0:
            for param in self.params:
                # There is no need in-place for gradient because it is only used to
                # update data then can be discarded.
                # But there is some operations such as regulization is helpful to be in-place
                param.grad = param.grad + self.weight_decay * param.data # must use in-place
                #param.grad.data.add_(self.weight_decay * param.data) # must use in-place

        if self.momentum != 0:
            for i, param in enumerate(self.params):
                if self.t > 1:
                    self.bt[i] = self.momentum * self.bt[i] + (1-self.dampening) * param.grad
                else:
                    self.bt[i] = param.grad

                param.grad.data = self.bt[i]


        for param in self.params:
            # Must be in-place because otherwise, param.data is new tensor
            # and the model's param.data is unchanged
            # To see affects, we will print id of param
            correctImp = True
            if correctImp:
                # Id after-before are same
                #print(f"----------")
                #print(f"Param.data id before changing {id(param.data)}")
                #print(f"Param id before changing {id(param)}")
                param.data.add_(-self.lr * param.grad)
                #print(f"Param.data id after changing {id(param.data)}")
                #print(f"Param id after changing {id(param)}")
            else:
                # Id after-before are different
                #print(f"----------")
                #print(f"Param.data id before changing {id(param.data)}")
                #print(f"Param id before changing {id(param)}")
                param.data = param.data - self.lr * param.grad # error
                #print(f"Param.data id after changing {id(param.data)}")
                #print(f"Param id after changing {id(param)}")



    def __repr__(self) -> str:
        return f"SGD(lr={self.lr}, momentum = {self.momentum}, weight_decay={self.weight_decay})"

tests.test_sgd(SGD)
# %%
class RMSprop:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-08,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
    ):
        '''Implements RMSprop.

        Like the PyTorch version, but assumes centered=False
            https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html

        '''
        params = list(params) # turn params into a list (because it might be a generator)
        self.params = params
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum

        self.t = 0
        self.v0 = []
        self.b0 = []
        for param in self.params:
            self.v0.append(t.zeros_like(param))
            self.b0.append(t.zeros_like(param))

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:

        for i, (param, v, b) in enumerate(zip(self.params, self.v0, self.b0)):
            # Each param, v, b is a reference
            grad = param.grad

            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param

            # We use assignment so we just **created** a new tensor
            v = self.alpha*v + (1-self.alpha) * t.pow(grad,2)
            # or you can use new_v = self.alpha*v 
            # so that you do not need to do self.v0[i] = v at the end

            if self.momentum >0:
                b = self.momentum * b + grad / (t.sqrt(v) + self.eps)
                param.add_(-self.lr * b) # Fine either, as param is a reference
                #self.params[i].add_(-self.lr * b) # This is fine
            else:
                param.add_(-self.lr * grad / (t.sqrt(v) + self.eps))
                #self.params[i].add_(-self.lr * grad / (t.sqrt(v) + self.eps))

            # Because for v, and b we already use assignemt, thus they are new tensors
            # therefore we need to reupdate
            self.v0[i] = v
            self.b0[i] = b

    def __repr__(self) -> str:
        return f"RMSprop(lr={self.lr}, eps={self.eps}, momentum={self.mu}, weight_decay={self.lmda}, alpha={self.alpha})"



if MAIN:
    tests.test_rmsprop(RMSprop)
# %%
a = [1,2,3]
b = [9,8,3]
for x,y in zip(a,b):
    x += 10
    y += 9

print(a)
print(b)

a = [t.rand(2), t.rand(2), t.rand(2)]
print(a)
for x in a:
    x = t.ones_like(x)
print(a)
# %%
class Adam:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
    ):
        '''Implements Adam.

        Like the PyTorch version, but assumes amsgrad=False and maximize=False
            https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
        '''
        params = list(params) # turn params into a list (because it might be a generator)
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.m = [t.zeros_like(param) for param in params]
        self.v = [t.zeros_like(param) for param in params]
        self.t = 0

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        self.t += 1

        for i, (param, m, v) in enumerate(zip(self.params, self.m, self.v)):
            grad = param.grad

            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param
            
            new_m = self.betas[0] * m + (1-self.betas[0]) * grad
            new_v = self.betas[1] * v + (1-self.betas[1]) * t.pow(grad,2)
            m_hat = new_m / (1 - self.betas[0]**self.t)
            v_hat = new_v / (1 - self.betas[1]**self.t)

            self.params[i] += (-self.lr * m_hat) / (t.sqrt(v_hat) + self.eps)

            self.v[i] = new_v
            self.m[i] = new_m

    def __repr__(self) -> str:
        return f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}, weight_decay={self.lmda})"



if MAIN:
    tests.test_adam(Adam)
# %%
class AdamW:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
    ):
        '''Implements Adam.

        Like the PyTorch version, but assumes amsgrad=False and maximize=False
            https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
        '''
        params = list(params) # turn params into a list (because it might be a generator)
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.m = [t.zeros_like(param) for param in params]
        self.v = [t.zeros_like(param) for param in params]
        self.t = 0

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        self.t += 1

        for i, (param, m, v) in enumerate(zip(self.params, self.m, self.v)):
            grad = param.grad

            self.params[i] +=  -self.weight_decay * self.lr * self.params[i]
            
            new_m = self.betas[0] * m + (1-self.betas[0]) * grad
            new_v = self.betas[1] * v + (1-self.betas[1]) * t.pow(grad,2)
            m_hat = new_m / (1 - self.betas[0]**self.t)
            v_hat = new_v / (1 - self.betas[1]**self.t)

            self.params[i] += (-self.lr * m_hat) / (t.sqrt(v_hat) + self.eps)

            self.v[i] = new_v
            self.m[i] = new_m

    def __repr__(self) -> str:
        return f"AdamW(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}, weight_decay={self.lmda})"



if MAIN:
    tests.test_adamw(AdamW)
# %%
def opt_fn(fn: Callable, xy: t.Tensor, optimizer_class, optimizer_hyperparams: dict, n_iters: int = 100):
    '''Optimize the a given function starting from the specified point.

    optimizer_class: one of the optimizers you've defined, either SGD, RMSprop, or Adam
    optimzer_kwargs: keyword arguments passed to your optimiser (e.g. lr and weight_decay)
    '''
    outputs = []
    parameters = [t.nn.Parameter(xy[0]), t.nn.Parameter(xy[1])]
    optimizers = optimizer_class(parameters, **optimizer_hyperparams)


    for _ in range(n_iters):
        outputs.append(t.Tensor([i.detach() for i in parameters]))

        optimizers.zero_grad()

        loss = fn(*parameters)
        loss.backward()

        optimizers.step()

    outputs = t.stack(outputs, dim=0)
    return outputs
if MAIN:
    points = []

    optimizer_list = [
        (SGD, {"lr": 0.03, "momentum": 0.99}),
        (RMSprop, {"lr": 0.02, "alpha": 0.99, "momentum": 0.8}),
        (Adam, {"lr": 0.2, "betas": (0.99, 0.99), "weight_decay": 0.005}),
    ]

    for optimizer_class, params in optimizer_list:
        xy = t.tensor([2.5, 2.5], requires_grad=True)
        xys = opt_fn(pathological_curve_loss, xy=xy, optimizer_class=optimizer_class, optimizer_hyperparams=params)
        points.append((xys, optimizer_class, params))

    plot_fn_with_points(pathological_curve_loss, points=points)
# %%
class SGD:

    def __init__(self, params, **kwargs):
        '''Implements SGD with momentum.

        Accepts parameters in groups, or an iterable.

        Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
            https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
        '''
        params = list(params)
        print(params)

        if isinstance(params[0], t.nn.Parameter):
            self.params = params
            self.lr = None
            self.momentum = [0] * len(params)
            self.dampening = [0] * len(params)
            self.weight_decay = [0] * len(params)

            if "lr" in kwargs.keys():
                self.lr = kwargs["lr"] * len(params)
            else:
                raise KeyError("lr must be set")
            
            if "momentum" in kwargs.keys():
                self.momentum = kwargs["momentum"] * len(params)
            if "weight_decay" in kwargs.keys():
                self.weight_decay = kwargs["weight_decay"] * len(params)

        elif isinstance(params[0], dict):
            self.params = []
            self.momentum = []
            self.lr = []
            self.weight_decay = []
            # Get the default lr, momentum, weigh_decay first
            lr = None
            momentum = 0
            weight_decay = 0
            param_set = set() # check that no parameters overlap
            # Do not raise lr because maybe inside each params, they already have lr
            if "lr" in kwargs.keys():
                lr = kwargs["lr"]
            if "momentum" in kwargs.keys():
                momentum = kwargs["momentum"]
            if "weight_decay" in kwargs.keys():
                weight_decay = kwargs["weight_decay"]

            for param_info in params:
                num_params = None
                if "params" in param_info.keys():
                    val = list(param_info["params"])
                    if len(param_set.intersection(val)) > 0:
                        raise KeyError("Duplicate parameters")
                    self.params.extend(val)
                    num_params = len(val)
                    param_set.update(val)
                else:
                    raise KeyError("No params in dict")

                if "lr" in param_info.keys():
                    self.lr.extend([param_info["lr"]] * num_params)
                else:
                    if lr == None:
                        raise KeyError("No lr in default keywords and dict")
                    else:
                        self.lr.extend([lr] * num_params)

                if "momentum" in param_info.keys():
                    self.momentum.extend([param_info["momentum"]] * num_params)
                else:
                    if momentum == None:
                        self.momentum.extend([0] * num_params)
                    else:
                        self.momentum.extend([momentum] * num_params)

                if "weight_decay" in param_info.keys():
                    self.weight_decay.extend([param_info["weight_decay"]] * num_params)
                else:
                    if weight_decay == None:
                        self.weight_decay.extend([0] * num_params)
                    else:
                        self.weight_decay.extend([weight_decay] * num_params)


        else:
            raise KeyError("Invalid keyword")

        self.t = 0
        self.b = [t.zeros_like(param) for param in self.params]
        
        print(params)
        print(kwargs)
        print(f"Len of params {self.params}, lr {self.lr} w {self.weight_decay}" \
            f" mom: {self.momentum}")
        print("----")

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        self.t += 1

        for i, param in enumerate(self.params):
            grad = param.grad
            if self.weight_decay[i] != 0:
                grad = grad + self.weight_decay[i] * param

            if self.momentum[i] != 0:
                if self.t > 1:
                    grad = self.momentum[i] * self.b[i] + grad

            param += -self.lr[i] * grad

            self.b[i] = grad


if MAIN:
    tests.test_sgd_param_groups(SGD)
# %%
def x(**kwargs):
    print(kwargs)
    print(kwargs['x'])

x(**{'y': 100})
# %%
