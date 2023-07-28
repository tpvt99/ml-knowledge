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
def conv1d_minimal_simple(x: Float[Tensor, "w"], weights: Float[Tensor, "kw"]) -> Float[Tensor, "ow"]:
    '''
    Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

    Simplifications: batch = input channels = output channels = 1.

    x: shape (width,)
    weights: shape (kernel_width,)

    Returns: shape (output_width,)
    '''
    input_width = x.size(0)
    weight_width = weights.size(0)

    output_width = input_width - weight_width+1

    ## expand the inputs
    x_stride = x.stride()[0]
    x_expanded = torch.as_strided(x, size=(output_width, weight_width), stride=(x_stride, x_stride))
    assert x_expanded.shape == (output_width, weight_width)

    # then we can either expand weights and doing element-wise multiplication
    # Option 1. 
    weight_expanded = torch.as_strided(weights, size=(output_width, weight_width), stride=(0, weights.stride()[0]))
    output1 = x_expanded * weight_expanded
    result1 = output1.sum(dim=-1)

    # or doing einsum
    # Option 2.
    result2 = einops.einsum(x_expanded, weights, 'o w, w -> o')

    assert torch.allclose(result1, result2)

    return result2


if MAIN:
    tests.test_conv1d_minimal_simple(conv1d_minimal_simple)
# %%
def conv1d_minimal(x: Float[Tensor, "b ic w"], weights: Float[Tensor, "oc ic kw"]) -> Float[Tensor, "b oc ow"]:
    '''
    Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    '''
    b, ic, w = x.shape
    oc, _, kw = weights.shape
    # Get output width, using formula
    ow = w - kw + 1

    # Stride of x
    stride_b, stride_ic, stride_w = x.stride()
    # Get the strided of x
    x_new_shape = (b, ic, ow, kw)
    x_new_stride = (stride_b, stride_ic, stride_w, stride_w)
    x_strided = x.as_strided(size = x_new_shape, stride = x_new_stride)

    # multiplication
    # For each input channels, it is summed togheter
    # And for each elements in width of weights * width of inputs, they are also summed.
    output = einops.einsum(x_strided, weights, 'b ic ow kw, oc ic kw -> b oc ow')

    return output


if MAIN:
    tests.test_conv1d_minimal(conv1d_minimal)
# %%
def conv2d_minimal(x: Float[Tensor, "b ic h w"], weights: Float[Tensor, "oc ic kh kw"]) -> Float[Tensor, "b oc oh ow"]:
    '''
    Like torch's conv2d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    b, ic, h, w = x.shape
    oc, _, kh, kw = weights.shape

    # Get output height and width
    oh = h - kh + 1
    ow = w - kw + 1

    # stride of x
    stride_b, stride_ic, stride_h, stride_w = x.stride()
    x_new_shape = (b, ic, oh, ow, kh, kw)
    x_new_stride = (stride_b, stride_ic, stride_h, stride_w,stride_h, stride_w)
    x_stride = x.as_strided(size = x_new_shape, stride=x_new_stride)

    # einsum
    output = einops.einsum(x_stride, weights, 'b ic oh ow kh kw, oc ic kh kw -> b oc oh ow')

    return output



if MAIN:
    tests.test_conv2d_minimal(conv2d_minimal)

# %%
def pad1d(x: t.Tensor, left: int, right: int, pad_value: float) -> t.Tensor:
    '''Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, width), dtype float32

    Return: shape (batch, in_channels, left + right + width)
    '''
    b, c, w = x.shape
    output = x.new_full(size=(b, c, w + left+right), fill_value=pad_value)
    output[..., left:w+left] = x

    return output


if MAIN:
    tests.test_pad1d(pad1d)
    tests.test_pad1d_multi_channel(pad1d)

# %%
def pad2d(x: t.Tensor, left: int, right: int, top: int, bottom: int, pad_value: float) -> t.Tensor:
    '''Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, height, width), dtype float32

    Return: shape (batch, in_channels, top + height + bottom, left + width + right)
    '''
    b, c, h, w = x.shape
    output = x.new_full(size=(b, c, h+top+bottom, w + left+right), fill_value=pad_value)
    output[..., top: h + top, left:w+left] = x

    return output


if MAIN:
    tests.test_pad2d(pad2d)
    tests.test_pad2d_multi_channel(pad2d)
# %%
def conv1d(
    x: Float[Tensor, "b ic w"], 
    weights: Float[Tensor, "oc ic kw"], 
    stride: int = 1, 
    padding: int = 0
) -> Float[Tensor, "b oc ow"]:
    '''
    Like torch's conv1d using bias=False.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    '''
    x = pad1d(x, left=padding, right=padding, pad_value = 0.0) 

    b, ic, w = x.shape
    oc, _, kw = weights.shape

    # Get output width
    # OH NO, I am still add 2*padding even after I pad. It is wrong
    ow = ((w - kw)//stride) + 1


    # stride of x
    stride_b, stride_ic, stride_w = x.stride()
    x_new_shape = (b, ic, ow, kw)
    x_new_stride = (stride_b, stride_ic, stride_w*stride , stride_w)
    x_stride = x.as_strided(size = x_new_shape, stride=x_new_stride)

    # einsum
    output = einops.einsum(x_stride, weights, 'b ic ow kw, oc ic kw -> b oc ow')
    return output


if MAIN:
    tests.test_conv1d(conv1d)
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

# Examples of how this function can be used:


if MAIN:
    for v in [(1, 2), 2, (1, 2, 3)]:
        try:
            print(f"{v!r:9} -> {force_pair(v)!r}")
        except ValueError:
            print(f"{v!r:9} -> ValueError")
# %%
def conv2d(
    x: Float[Tensor, "b ic h w"], 
    weights: Float[Tensor, "oc ic kh kw"], 
    stride: IntOrPair = 1, 
    padding: IntOrPair = 0
) -> Float[Tensor, "b oc oh ow"]:
    '''
    Like torch's conv2d using bias=False

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    pad_h, pad_w = force_pair(padding)
    s_h, s_w = force_pair(stride)
    x = pad2d(x, left = pad_w, right = pad_w, top = pad_h, bottom = pad_h, pad_value=0)

    b, ic, h, w = x.shape
    oc, _, kh, kw = weights.shape

    # Get output height and width
    oh = (h - kh)//s_h + 1
    ow = (w - kw)//s_w + 1

    # stride of x
    stride_b, stride_ic, stride_h, stride_w = x.stride()
    x_new_shape = (b, ic, oh, ow, kh, kw)
    x_new_stride = (stride_b, stride_ic, stride_h * s_h, stride_w * s_w, stride_h, stride_w)
    x_stride = x.as_strided(size = x_new_shape, stride=x_new_stride)

    # einsum
    output = einops.einsum(x_stride, weights, 'b ic oh ow kh kw, oc ic kh kw -> b oc oh ow')

    return output


if MAIN:
    tests.test_conv2d(conv2d)
# %%
def maxpool2d(
    x: Float[Tensor, "b ic h w"], 
    kernel_size: IntOrPair, 
    stride: Optional[IntOrPair] = None, 
    padding: IntOrPair = 0
) -> Float[Tensor, "b ic oh ow"]:
    '''
    Like PyTorch's maxpool2d.

    x: shape (batch, channels, height, width)
    stride: if None, should be equal to the kernel size

    Return: (batch, channels, output_height, output_width)
    '''
    padh, padw = force_pair(padding)
    kh, kw = force_pair(kernel_size)
    sh, sw = kh, kw
    if stride:
        sh, sw = force_pair(stride)
    # Should not pad 0 but minimum so it wont affect max pool
    x = pad2d(x, left=padw, right=padw, top=padh, bottom=padh, pad_value=-torch.inf)

    b, ic, h, w = x.shape

    # Output h w
    oh = (h - kh)//sh + 1
    ow = (w - kw)//sw + 1

    stride_b, stride_ic, stride_h, stride_w = x.stride()
    # Build new x_strided
    x_new_shape = (b, ic, oh, ow, kh, kw)
    x_new_stride = (stride_b, stride_ic, stride_h * sh, stride_w * sw, stride_h, stride_w)
    x_stride = x.as_strided(size = x_new_shape, stride = x_new_stride)

    output = einops.reduce(x_stride, 'b ic oh ow kh kw -> b ic oh ow', 'max')
    output2 = torch.amax(x_stride, dim=(-2, -1))

    assert torch.allclose(output, output2)

    return output


if MAIN:
    tests.test_maxpool2d(maxpool2d)
# %%
