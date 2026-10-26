#%%
import numpy as np
from numpy.lib.stride_tricks import as_strided
from collections import namedtuple
import torch as t
from jaxtyping import Float
from torch import Tensor
import tests
import sys
from pathlib import Path
import einops

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

#%%
# follow https://medium.com/data-science/advanced-numpy-master-stride-tricks-with-25-illustrated-exercises-923a9393ab20#6c04
# Number of strides is byte between element
#1/ slice the first 3
x = np.asarray(range(1,26), np.int8).reshape(5, 5)
print(x)
print(as_strided(x, shape=(3,), strides=(1,)))

#2/ slice the first 8
x = np.asarray(range(1,26), np.int8).reshape(5, 5)
print(x)
print(as_strided(x, shape=(8,), strides=(1,)))

#3/ flatten 2d array
x = np.asarray(range(1,26), np.int16).reshape(5, 5)
print(x.strides)
print(as_strided(x, shape=(25,), strides=(2,)))

# 4/ skip every other element
x = np.asarray(range(1, 26), dtype=np.int8).reshape(5,5)
print(x.strides)
print(as_strided(x, shape=(3,), strides=(2,)))

# 5/ stride the first column
x = np.asarray(range(1, 26), dtype=np.int64).reshape(5,5)
print(x.strides)
print(as_strided(x, shape=(4,), strides=(40,)))

# 6/ slice diagonal
x = np.asarray(range(1, 26), dtype=np.int64).reshape(5,5)
print(x.strides)
print(as_strided(x, shape=(5,), strides=(48,)))

# 7/ repeate the first element
x = np.asarray(range(1, 26), dtype=np.int64).reshape(5,5)
print(x.strides)
print(as_strided(x, shape=(5,), strides=(0,)))

# 8/ simple 2d slicing
x = np.asarray(range(1, 26), dtype=np.int64).reshape(5,5)
print(x.strides)
print(as_strided(x, shape=(3,4), strides=(40,8)))
# you can think as for axis=0, 3 elements, each element 40 bytes apart
# and for axis =1, 4 elements, each element is 8 bytes apart

# 9/ slicing a zigzag
x = np.asarray(range(1, 26), dtype=np.int64).reshape(5,5)
print(x.strides)
print(as_strided(x, shape=(4,2), strides=(48,8)))
# it is easy, for axis = 0, 4 elements, each element apart 48 bytes
# for axis=1, 2 elements, each element is only 8 bytes apart

# 10/ sparse slicing
x = np.asarray(range(1, 26), dtype=np.int64).reshape(5,5)
print(x.strides)
print(as_strided(x, shape=(3,3), strides=(10*8,2*8)))
# for axis=0, 1 to 11 is 10 elements away so 10*8 bytes
# for axis=1, 1 to 3 is 2 elements away so 2*8 bytes

# 11/ transpose 2d array
x = np.asarray(range(1, 26), dtype=np.int8).reshape(5,5)
print(x.strides)
print(as_strided(x, shape=(3,3), strides=(1,5*1)))
# for tranpose, axis=0, 1-2 is 1 element away so 1*1byte,
# for axis=1, 1-6 is 5 element away so 5*1 byte

#%%
# 12/ repeat the first column 4 times
x = np.asarray(range(1, 26), dtype=np.int32).reshape(5,5)
print(x.strides)
print(as_strided(x, shape=(5,4), strides=(5*4,0)))

# 13/ rehape 1d array to 2d array
x = np.asarray(range(1,13), np.int64)
print(x)
print(x.strides)
as_strided(x, shape=(4,3), strides=(3*8, 1*8))

# 14/ slide a 1d window
x = np.asarray(range(1,11), np.int8)
print(x.strides)
as_strided(x, shape=(8,3), strides=(1,1))

# 15/ slide a 2d window then flatten
x = np.asarray([0,1,10,11,20,21,30,31,40,41,50,51], np.int8).reshape(6,2)
print(x.strides)
as_strided(x, shape=(4,6), strides=(1*2,1))
# for axis=0 (each row) from 0-10 is 2 elements so 2*1byte = 2
# for axis=1 (each column) from 0-1, 1-10 is 1 element apart so 1*1 =1

# 16/ collapse an axis from 3d array
x = np.asarray(range(1,13), np.int8).reshape(3,2,2)
print(x.strides)
as_strided(x, shape=(3,4), strides=(4,1))
# %%
# 17/ 2 corners
x = np.asarray(range(1,26), np.int16).reshape(5,5)
as_strided(x, shape=(2,2,2), strides=(15*2, 5*2, 1*2))

# 18/ satggered slicing
x = np.asarray(range(1,26), np.int8).reshape(5,5)
as_strided(x, shape=(2,2,3), strides=(10*1, 6*1, 1*1))

# 19/ repeat a 2d array
x = np.asarray(range(1,26), np.int16).reshape(5,5)
as_strided(x, shape=(3,2,4), strides=(0*2, 5*2, 1*2))

# 20/ 3d transpose
x = np.asarray(range(1,13), np.int32).reshape(3,2,2)
as_strided(x, shape=(3,2,2), strides=(4*4, 1*4, 2*4))

# 21/ slide a 2d window
x = np.asarray(range(1,21), np.int64).reshape(4,5)
as_strided(x, shape=(3,2,5), strides=(5*8, 5*8, 1*8))

# 22/ reshape 1d array to 3d array
x = np.asarray(range(1,13), np.int8)
as_strided(x, shape=(2,2,3), strides=(6*1, 3*1, 1*1))

# %%
# 23/ slide a 2d receptive field for convolutions
x = np.asarray(range(1,26), np.int8)
as_strided(x, shape=(2,2,3,3), strides=(10*1, 2*1, 5*1, 1*1))

# 24/ repeat a 3d tensor
x = np.asarray(range(1,13), np.int64).reshape(2,2,3)
as_strided(x, shape=(2,2,2,3), strides=(6*8, 0*8, 3*8, 1*8))

# 25/ reshape 1d to 4d
x = np.asarray(range(1,17), np.int64)
as_strided(x, shape=(2,2,2,2), strides=(8*8, 4*8, 2*8, 1*8))

# %%
# Now is for the lesson
test_input = t.tensor(
    [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19],
    ],
    dtype=t.float,
)
print(test_input.stride())

TestCase = namedtuple("TestCase", ["output", "size", "stride"])

test_cases = [
    # Example 1
    TestCase(
        output=t.tensor([0, 1, 2, 3]),
        size=(4,),
        stride=(1,),
    ),
    # Example 2
    TestCase(
        output=t.tensor([[0, 2], [5, 7]]),
        size=(2, 2),
        stride=(5, 2),
    ),
    # Start of exercises (you should fill in size & stride for all 6 of these):
    TestCase(
        output=t.tensor([0, 1, 2, 3, 4]),
        size=(5,),
        stride=(1,),
    ),
    TestCase(
        output=t.tensor([0, 5, 10, 15]),
        size=(4,),
        stride=(5,),
    ),
    TestCase(
        output=t.tensor([[0, 1, 2], [5, 6, 7]]),
        size=(2,3),
        stride=(5,1),
    ),
    TestCase(
        output=t.tensor([[0, 1, 2], [10, 11, 12]]),
        size=(2,3),
        stride=(10,1),
    ),
    TestCase(
        output=t.tensor([[0, 0, 0], [11, 11, 11]]),
        size=(2,3),
        stride=(11,0),
    ),
    TestCase(
        output=t.tensor([0, 6, 12, 18]),
        size=(4,),
        stride=(6,),
    ),
]

for i, test_case in enumerate(test_cases):
    if (test_case.size is None) or (test_case.stride is None):
        print(f"Test {i} failed: attempt missing.")
    else:
        actual = test_input.as_strided(size=test_case.size, stride=test_case.stride)
        if (test_case.output != actual).any():
            print(f"Test {i} failed\n  Expected: {test_case.output}\n  Actual: {actual}")
        else:
            print(f"Test {i} passed!")

# %%
def as_strided_trace(mat: Float[Tensor, "i j"]) -> Float[Tensor, ""]:
    """
    Returns the same as `torch.trace`, using only `as_strided` and `sum` methods.
    """
    stride = mat.stride()
    return mat.as_strided(size=(mat.size(0),), stride=(stride[0]+stride[1],)).sum()


tests.test_trace(as_strided_trace)

def as_strided_mv(mat: Float[Tensor, "i j"], vec: Float[Tensor, " j"]) -> Float[Tensor, " i"]:
    """
    Returns the same as `torch.matmul`, using only `as_strided`, `sum`, and elementwise multiplication `*` methods.
    """
    assert mat.size(1) == vec.size(0), "Must match last dimension"
    # Turn vec to [[---j----], [---j----], ...]
    vec_repeat = vec.as_strided(size=(mat.size(0), mat.size(1)), stride=(0, vec.stride()[0]))
    matmul = mat * vec_repeat # `*` is always elementwise multiplication
    return matmul.sum(dim=-1)
    
tests.test_mv(as_strided_mv)
tests.test_mv2(as_strided_mv)

def as_strided_mm(matA: Float[Tensor, "i j"], matB: Float[Tensor, "j k"]) -> Float[Tensor, "i k"]:
    """
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    """
    sizeA = matA.shape
    sizeB = matB.shape
    assert len(sizeA) == 2 and len(sizeB) == 2, "Shoud have 2d shape"
    assert sizeA[1] == sizeB[0], "Must match dimension"

    strideA = matA.stride()
    strideB = matB.stride()

    # Hmm probably i need to broadcast matA and increase dim of matB
    # doing this broadcast makes me feel very strange if they add dim at last dim, for example, given matrix shape (2,5), it is naturally to extend (1,2,5) but when you try to extend it to (2,5,1) it rearange in a very different way i never thought off but somehow it works so let's do it. However with this way the matB extend the first dim like (1, *sizeB) so you feel naturally with B but weird with A.
    extended_A = matA.as_strided(size=(*sizeA, sizeB[1]), stride=(strideA[0], strideA[1], 0))
    extended_B = matB.as_strided(size=(sizeA[0], *sizeB), stride=(0, strideB[0], strideB[1]))
    mul = extended_A * extended_B
    sum_mat = mul.sum(dim=1)
    return sum_mat

    # Okie here i extend B as (sizeB[1], sizeA[0], sizeA[1]) so when element wise multiplication
    # we have dim [sizeB[1], sizeA[0], sizeA[1]] then sum over sizeA[0], but later need transpose
    extended_B = matB.as_strided(size=(sizeB[1], *sizeA), stride=(matB.stride()[1], 0, matB.stride()[0]))
    mul = matA * extended_B # multiply
    sum_mat = mul.sum(dim=-1) # then sum to have shape (sizeB[1], sizeA[0])
    # we do transpose and totaly can use stride
    tranpose_mat = sum_mat.as_strided(size=(sizeA[0], sizeB[1]), stride=(sum_mat.stride()[1], sum_mat.stride()[0]))
    return tranpose_mat

tests.test_mm(as_strided_mm)
tests.test_mm2(as_strided_mm)
# %%
x = t.tensor(range(1,11)).reshape(2,5) 
y = t.tensor(range(1,21)).reshape(5,4)
print(x)
print(y)
print(x @ y)
# first method need transpose but no need manually add dim for broadcast to work
temp = y.as_strided(size=(4,2,5), stride=(y.stride()[1], 0, y.stride()[0]))
print(temp)
sum_mat = (x * temp).sum(-1)
print(sum_mat)
print(sum_mat.stride())
sum_mat.as_strided(size=(x.shape[0], y.shape[1]), stride=(sum_mat.stride()[1], sum_mat.stride()[0]))
# second method no need transpose but need add dim
print('----------')
temp = y.as_strided(size=(2,5,4), stride=(0, y.stride()[0], y.stride()[1]))
print(temp)
print(x[..., None].broadcast_to(2,5,4)) # omg this is feel like i need stride but it can do itself
print(x.as_strided(size=(2,5,4), stride=(x.stride()[0], x.stride()[1], 0)))
print(x[..., None] * y)
# %%
# Conv1d
def conv1d_minimal_simple(
    x: Float[Tensor, "width"], weights: Float[Tensor, "kernel_width"]
) -> Float[Tensor, "output_width"]:
    """
    Like torch's conv1d using bias=False and all other keyword arguments left at default values.

    Simplifications: batch = input channels = output channels = 1. 
    """
    assert len(x.shape) == 1, "Supporting 1d conv only"
    assert len(weights.shape) == 1, "Supporting 1d conv only"

    output_width = x.size(0) - weights.size(0)+1
    # Create new array with [output_width, weight.size]
    mat = x.as_strided(size=(output_width, weights.size(0)), stride=(x.stride()[0], x.stride()[0]))
    # Then multiply mat with weight using einsum
    return einops.einsum(mat, weights, "a k, k -> a")

tests.test_conv1d_minimal_simple(conv1d_minimal_simple)

def conv1d_minimal(
    x: Float[Tensor, "batch in_channels width"],
    weights: Float[Tensor, "out_channels in_channels kernel_width"],
) -> Float[Tensor, "batch out_channels output_width"]:
    """
    Like torch's conv1d using bias=False and all other keyword arguments left at default values.
    """
    batch, in_channels, in_width = x.shape
    out_channels, _, kernel_width = weights.shape

    out_width = in_width - kernel_width + 1
    batch_stride, inc_stride, inw_stride = x.stride()

    # Create new array with shape [batch, in_channels, out_width, kernel_width]
    mat = x.as_strided(size=(batch, in_channels, out_width, kernel_width), stride=(batch_stride, inc_stride, inw_stride, inw_stride))
    # Then multiply
    # Batch is not summed over, out_channels is not summed over
    # what is summed and multiply is kernel_width, in_channels because it exist in input and lack in output ()
    return einops.einsum(mat, weights, 
            "batch in_channels out_width kernel_width, out_channels in_channels kernel_width -> batch out_channels out_width")

tests.test_conv1d_minimal(conv1d_minimal)

def conv2d_minimal(
    x: Float[Tensor, "batch in_channels height width"],
    weights: Float[Tensor, "out_channels in_channels kernel_height kernel_width"],
) -> Float[Tensor, "batch out_channels height_padding width_padding"]:
    """
    Like torch's conv2d using bias=False and all other keyword arguments left at default values.
    """
    b, ic, h, w = x.shape
    oc, ic2, kh, kw = weights.shape

    assert ic == ic2, "in_channels of x and weights must match"
    # Get strides
    s_b, s_ic, s_h, s_w = x.stride()
    # Get expected output
    out_h = h - kh + 1
    out_w = w - kw + 1

    # make it stride
    x_strided_size = (b, ic, out_h, out_w, kh, kw)
    x_strided_stride = (s_b, s_ic, s_h, s_w, s_h, s_w)
    x_strided = x.as_strided(size=x_strided_size, stride=x_strided_stride)

    # Multiply and sum using einops
    return einops.einsum(x_strided, weights, 
                         "b ic out_h out_w kh kw, oc ic kh kw -> b oc out_h out_w")


tests.test_conv2d_minimal(conv2d_minimal)

def pad1d(
    x: Float[Tensor, "batch in_channels width"], left: int, right: int, pad_value: float
) -> Float[Tensor, "batch in_channels width_padding"]:
    """Return a new tensor with padding applied to the edges."""
    # Get shape
    b, inc, width = x.shape
    padded_x = x.new_full(size=(b, inc, width+left+right), fill_value=pad_value)

    padded_x[..., left:left+width] = x
    #padded_x[..., :left] = pad_value #Because we use pad_value for fill_value so no need reassign
    #padded_x[..., left+width:] = pad_value

    return padded_x

tests.test_pad1d(pad1d)
tests.test_pad1d_multi_channel(pad1d)

def pad2d(
    x: Float[Tensor, "batch in_channels height width"],
    left: int,
    right: int,
    top: int,
    bottom: int,
    pad_value: float,
) -> Float[Tensor, "batch in_channels height_padding width_padding"]:
    """Return a new tensor with padding applied to the width & height dimensions."""
    b, inc, h, w = x.shape
    output = x.new_full(size=(b, inc, top + h + bottom, left + w + right), fill_value=pad_value)
    output[..., top:top+h,left:left+w] = x
    return output


tests.test_pad2d(pad2d)
tests.test_pad2d_multi_channel(pad2d)

def conv1d(
    x: Float[Tensor, "batch in_channels width"],
    weights: Float[Tensor, "out_channels in_channels kernel_width"],
    stride: int = 1,
    padding: int = 0,
) -> Float[Tensor, "batch out_channels width"]:
    """
    Like torch's conv1d using bias=False.
    """
    x = pad1d(x, left=padding, right=padding, pad_value=0)

    # Get shape
    b, inc, w = x.shape
    outc, inc2, kw = weights.shape
    assert inc == inc2, "x and weight's in channel must be the same"

    # Output shape
    out_width = int(np.floor((w - kw) / stride) + 1)
    # Find stride
    s_b, s_in, s_w = x.stride()
    stride_size = (b, inc, out_width, kw)
    stride_stride = (s_b, s_in, s_w * stride, s_w)

    x_stride = x.as_strided(size=stride_size, stride=stride_stride)
    return einops.einsum(x_stride, weights, "b inc outw kw, outc inc kw -> b outc outw")


tests.test_conv1d(conv1d)
# %%
IntOrPair = int | tuple[int, int]
Pair = tuple[int, int]

def force_pair(v: IntOrPair) -> Pair:
    """Convert v to a pair of int, if it isn't already."""
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(v)
        return (int(v[0]), int(v[1]))
    elif isinstance(v, int):
        return (v, v)
    raise ValueError(v)


# Examples of how this function can be used:
for v in [(1, 2), 2, (1, 2, 3)]:
    try:
        print(f"{v!r:9} -> {force_pair(v)!r}")
    except ValueError:
        print(f"{v!r:9} -> ValueError")

def conv2d(
    x: Float[Tensor, "batch in_channels height width"],
    weights: Float[Tensor, "out_channels in_channels kernel_height kernel_width"],
    stride: IntOrPair = 1,
    padding: IntOrPair = 0,
) -> Float[Tensor, "batch out_channels height width"]:
    """
    Like torch's conv2d using bias=False.
    """
    padding = force_pair(padding)
    pad_height, pad_width = padding
    stride = force_pair(stride)
    stride_height, stride_width = stride

    x_padded = pad2d(x, left=pad_width, right=pad_width, top=pad_height, bottom=pad_height)


tests.test_conv2d(conv2d)
# %%
