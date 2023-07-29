#%%

import os
import sys
import numpy as np
import einops
from typing import Union, Optional, Tuple
import torch as t
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
import functools
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm.notebook import tqdm

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part2_cnns"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, line, bar
from part2_cnns.utils import display_array_as_img, display_soln_array_as_img
import part2_cnns.tests as tests

MAIN = __name__ == "__main__"
# %%

IntOrPair = Union[int, Tuple[int, int]]
Pair = Tuple[int, int]

def force_pair(v: IntOrPair) -> Pair:
    '''Convert v to a pair of int, if it isn't already.'''
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(v)
        return (int(v[0]), int(v[1]))
    elif isinstance(v, int):
        return (v, v)
    raise ValueError(v)

def pad2d(x: t.Tensor, left: int, right: int, top: int, bottom: int, pad_value: float) -> t.Tensor:
    '''Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, height, width), dtype float32

    Return: shape (batch, in_channels, top + height + bottom, left + width + right)
    '''
    b, c, h, w = x.shape
    output = x.new_full(size=(b, c, h+top+bottom, w + left+right), fill_value=pad_value)
    output[..., top: h + top, left:w+left] = x

    return output


### MaxPool2d
class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: IntOrPair, stride: Optional[IntOrPair] = None, 
                 padding: IntOrPair = 1):
        super().__init__()
        
        self.kh, self.kw = force_pair(kernel_size)
        self.padh, self.padw = force_pair(padding)
        self.sh, self.sw = self.kh, self.kw
        if stride:
            self.sh, self.sw = force_pair(stride)

    def forward(self, x: Float[t.Tensor, "batch ic h w"]) -> t.Tensor:
        '''
            Call the functional version of maxpool2d
        '''
        x = pad2d(x, left=self.padw, right=self.padw, top=self.padh, bottom=self.padh, pad_value=-torch.inf)

        b, ic, h, w = x.shape

        # Output h w
        oh = (h - self.kh)//self.sh + 1
        ow = (w - self.kw)//self.sw + 1

        stride_b, stride_ic, stride_h, stride_w = x.stride()
        # Build new x_strided
        x_new_shape = (b, ic, oh, ow, self.kh, self.kw)
        x_new_stride = (stride_b, stride_ic, stride_h * self.sh, stride_w * self.sw, stride_h, stride_w)
        x_stride = x.as_strided(size = x_new_shape, stride = x_new_stride)

        output = einops.reduce(x_stride, 'b ic oh ow kh kw -> b ic oh ow', 'max')
        #output2 = torch.amax(x_stride, dim=(-2, -1))

        return output

    def extra_repr(self) -> str:
        '''
            Add additional information to the string representation of this class
        '''
        #return f"kernel_size={self.kh, self.kw} stride={self.sh, self.sw}, padding={self.padh, self.padw})"
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["kh", "kw", "padh", "padw", "sh", "sw"]])

if MAIN:
    tests.test_maxpool2d_module(MaxPool2d)
    m = MaxPool2d(kernel_size=3, stride=2, padding=1)
    print(f"Manually verify that this is an informative repr: {m}")
# %%

class ReLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        # Option 1 using torch.where
        output1 = torch.where(x > 0, x, 0)

        # Option 2 using mask
        mask = (x > 0).int()
        output2 = mask * x

        # Option3 using maximum
        output3 = torch.maximum(x, torch.tensor(0.0))

        assert torch.allclose(output1, output2)
        assert torch.allclose(output1, output3)

        return output1


if MAIN:
    tests.test_relu(ReLU)
# %%

class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: t.Tensor) -> t.Tensor:
        '''
            Flatten out dimensions from start_dim to end_dim, inclusive of both
        '''
        # Option 1, using rearrange
        string_list = [f"a{i}" for i in range(input.ndim)]
        input_pattern = " ".join(string_list) # "a1 a2 a3 a4 a5"
        output_pattern = ""
        start_dim = self.start_dim if (self.start_dim >= 0) else input.ndim + self.start_dim
        end_dim = self.end_dim if (self.end_dim >= 0) else input.ndim + self.end_dim
        for i in range(input.ndim):
            if i ==start_dim:
                output_pattern += "(" + string_list[i] + " "
            elif i==end_dim:
                output_pattern += string_list[i] + ")"
            else:
                output_pattern += string_list[i] + " "

        output = einops.rearrange(input, f"{input_pattern} -> {output_pattern}")

        ## Option 2. Usingg reshape and functools.reduce
        shape = list(input.shape)
        end_dim = self.end_dim+1 if self.end_dim >= 0 else input.ndim + self.end_dim + 1
        condensed_num = functools.reduce(lambda x, y : x*y, shape[self.start_dim: end_dim])
        condensed_shape = shape[0: self.start_dim] + [condensed_num] + shape[end_dim:]
        output2 = torch.reshape(input, shape=condensed_shape)

        assert torch.allclose(output, output2)
        
        return output
    
    def extra_repr(self) -> str:
        return ", ".join([f"{key}: {getattr(self,key)}" for key in ["start_dim", "end_dim"]])

if MAIN:
    tests.test_flatten(Flatten)
# %%
f = Flatten(10,20)
print(f)
# %%
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        '''
        A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        '''
        super().__init__()
        self.weight = torch.zeros(size=(out_features, in_features), dtype=torch.float)
        self.weight.uniform_(-1/torch.sqrt(torch.tensor(in_features)), 1/torch.sqrt(torch.tensor(in_features)))
        self.weight = nn.Parameter(self.weight)
        
        self.bias = None
        if bias:
            self.bias = torch.zeros(size=(out_features,), dtype=torch.float)
            self.bias.uniform_(-1/torch.sqrt(torch.tensor(in_features)), 1/torch.sqrt(torch.tensor(in_features)))
            self.bias = nn.Parameter(self.bias)

    def forward(self, x: t.Tensor)-> t.Tensor:
        print(x.shape)
        output = einops.einsum(self.weight, x, 'out in, batch in -> batch out')
        if self.bias != None:
            output = output + self.bias

        return output

    def extra_repr(self) -> str:
        return ", ".join([f"{key}:{getattr(self, key)}" for key in ["weight", "bias"]])

if MAIN:
    tests.test_linear_forward(Linear)
    tests.test_linear_parameters(Linear)
    tests.test_linear_no_bias(Linear)
# %%
a = torch.tensor(10.0)
a.uniform_(-1.0,2.0)
del t
# %%
class Conv2d(nn.Module):
    def __init__(self, in_channels: int, 
                 out_channels:int,
                 kernel_size = IntOrPair,
                 stride: IntOrPair = 1,
                 padding: IntOrPair = 0):
        '''
        Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.
        '''
        super().__init__()
        self.kh, self.kw = force_pair(kernel_size)
        self.padh, self.padw = force_pair(padding)
        self.sh, self.sw = force_pair(stride)
        self.in_channels = in_channels
        self.out_channels = out_channels

        sf = np.sqrt(1/(in_channels * self.kh * self.kw))

        weight = sf * (2*torch.rand(size=(self.out_channels, self.in_channels, self.kh, self.kw)) - 1)
        self.weight = nn.Parameter(weight)

        bias = sf * (2*torch.rand(size=(self.out_channels,)) - 1)
        self.bias = nn.Parameter(bias)

    def forward(self, x: Float[t.Tensor, "batch c h w"]) -> t.Tensor:
        '''
            Apply the functional conv2d you wrote earlier
        '''
        x = pad2d(x, left = self.padw, right = self.padw, top = self.padh, bottom = self.padh, pad_value=0)

        b, ic, h, w = x.shape
        oc, _, kh, kw = self.weight.shape

        # Get output height and width
        oh = (h - kh)//self.sh + 1
        ow = (w - kw)//self.sw + 1

        # stride of x
        stride_b, stride_ic, stride_h, stride_w = x.stride()
        x_new_shape = (b, ic, oh, ow, kh, kw)
        x_new_stride = (stride_b, stride_ic, stride_h * self.sh, stride_w * self.sw, stride_h, stride_w)
        x_stride = x.as_strided(size = x_new_shape, stride=x_new_stride)

        # einsum
        print(f"x max {torch.max(x_stride)} x min {torch.min(x_stride)}")
        print(f"w max {torch.max(self.weight)} r min {torch.min(self.weight)}")
        print(f"b max {torch.max(self.bias)} b min {torch.min(self.bias)}")
        output = einops.einsum(x_stride, self.weight, 'b ic oh ow kh kw, oc ic kh kw -> b oc oh ow')
        print(f"output max {torch.max(output)} output min {torch.min(output)}")
        #output = output + self.bias.view(-1, 1, 1)
        print(f"output max {torch.max(output)} output min {torch.min(output)}")

        return output
    
    def extra_repr(self) -> str:
        return f"kernel_size: ({self.kh, self.kw}), padding: ({self.padh, self.padw})" \
                f"stride: ({self.sh, self.sw}) out_c {self.out_channels} in_c {self.in_channels}"
    
if MAIN:
    tests.test_conv2d_module(Conv2d)
# %%

x = torch.randn(3, 3, 7, 7)
my_conv = Conv2d(in_channels=3, out_channels=5, kernel_size=3, stride=1, padding=0)
torch_conv = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3, stride=1, padding=0)
my_y = my_conv(x)
torch_y = torch_conv(x)
print(my_conv.weight.var())
print(my_conv.bias.var())
print(my_conv.bias.shape)
print(torch_conv.weight.var())
print(torch_conv.bias.var())
print(torch_conv.bias.shape)
print(my_y)
print(torch_y)
# %%
