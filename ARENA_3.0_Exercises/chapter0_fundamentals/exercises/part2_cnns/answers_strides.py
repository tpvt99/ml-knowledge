#%%
import numpy as np
from numpy.lib.stride_tricks import as_strided
from collections import namedtuple, OrderedDict
import torch as t
from jaxtyping import Float, Int
from torch import Tensor
import tests
import sys
from pathlib import Path
import einops
import torch.nn as nn

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

    # Step 1. pad
    x_padded = pad2d(x, left=pad_width, right=pad_width, top=pad_height, bottom=pad_height, pad_value=0)

    # step 2. get size
    b, in_c, h, w = x_padded.shape
    out_c, in_c2, kh, kw = weights.shape
    assert in_c == in_c2, "x and weights must have same input channel"

    s_b, s_inc, s_h, s_w = x_padded.stride()

    # step 3. get expected output shape
    out_w = (w - kw) // stride_width + 1
    out_h = (h - kh) // stride_height + 1
    x_stride_shape = (b, in_c, out_h, out_w, kh, kw)
    x_stride_stride = (s_b, s_inc, stride_height * s_h, stride_width * s_w, s_h, s_w)
    x_strided = x_padded.as_strided(size = x_stride_shape, stride = x_stride_stride)

    return einops.einsum(x_strided, weights, "b inc h w kh kw, outc inc kh kw -> b outc h w")

tests.test_conv2d(conv2d)

def maxpool2d(
        x: Float[Tensor, "batch in_channels height width"],
        kernel_size: IntOrPair,
        stride: IntOrPair | None = None,
        padding: IntOrPair = 0
) -> Float[Tensor, "batch out_channels height width"]:
    """
    Like PyTorch's maxpool2d. If stride is None, should be equal to kernel size 
    """

    stride_h, stride_w = force_pair(stride)
    pad_h, pad_w = force_pair(padding)
    kernel_h, kernel_w = force_pair(kernel_size)

    # Pad
    x_padded = pad2d(x, pad_w, pad_w, pad_h, pad_h, 0)

    # then getting size and stride
    b, in_c, h, w = x_padded.shape
    s_b, s_inc, s_h, s_w = x_padded.stride()

    # as_strides's shape and stride
    out_w = (w - kernel_w) // stride_w + 1
    out_h = (h - kernel_h) // stride_h + 1
    x_stride_size = (b, in_c, out_h, out_w, kernel_h, kernel_w)
    x_stride_stride = (s_b, s_inc, stride_h * s_h, stride_w * s_w, s_h, s_w)
    x_stride = x_padded.as_strided(size=x_stride_size, stride=x_stride_stride)

    return einops.reduce(x_stride, "b inc outh outw kh kw -> b inc outh outw", lambda tensor, dims : t.amax(tensor, dim=dims))

tests.test_maxpool2d(maxpool2d)

# %%
# Now we will do fully Resnet from our manually written module

class BatchNorm2d(nn.Module):
    # The type hints below aren't functional, just for documentation
    running_mean: Float[Tensor, "num_features"]
    running_var: Float[Tensor, "num_features"]
    num_batches_tracked: Int[Tensor, ""] # scalar Tensor

    def __init__(self, num_features: int, eps=1e-5, momentum=0.1):
        """
        Like nn.BatchNorm2d with track_running_stats=True and affine=True
        Name the learnable affine parameters `weight` and `bias` in that order 
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = nn.Parameter(t.ones(num_features))
        self.bias = nn.Parameter(t.zeros(num_features))

        self.register_buffer("running_mean", t.zeros(num_features))
        self.register_buffer("running_var", t.ones(num_features))
        self.register_buffer("num_batches_tracked", t.tensor(0))

    def forward(self, x: Float[Tensor, "batch channels height width"]) \
            -> Float[Tensor, "batch channels height width"]:
        """
        Normalize each channel.
        In training mode, normalize using the biased variance `x.var(..., correction=0)`
        but update `running_var` using the unbiases variance `x.var(..., correction=1)`
        May find it useful to use the argument `keepdim`

        x: shape (batch, channels, height, width)
        return shape (batch, channels, height, width) 
        """
        if self.training:
            # step 1.a find mean and variance of mini batch, correction=0
            mean: Float[Tensor, "1 channels 1 1"] = x.mean(dim=[0,2,3], keepdim=True)
            var: Float[Tensor, "1 channels 1 1"] = x.var(dim=[0,2,3], correction=0, keepdim=True)

            # step 1.b need to calculate running average, var using momentum
            ## OKIE, here is bug I made, I thought running_mean.requires_grad = False always
            # because it is a buffer, but because we use mean which has require_grad=True
            # (x should have require_grad=False but it is from another layer such as Linear so
            # after that layer it multiply weight + bias which grad=True thus x.require_grad=True also)
            # so we need to wrap it in a no grad
            with t.no_grad():
                self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * mean.squeeze()
                unbiased_var: Float[Tensor, "channels"] = x.var(dim=[0,2,3], correction=1)
                self.running_var = (1-self.momentum) * self.running_var + self.momentum * unbiased_var
                self.num_batches_tracked += t.tensor(1)
        else:
            # for inference, use buffer mean and variance to normalize
            mean = einops.rearrange(self.running_mean, "c -> 1 c 1 1")
            var = einops.rearrange(self.running_var, "c -> 1 c 1 1")

        # step 2. normalize
        assert mean.shape == (1, x.shape[1], 1, 1)
        assert var.shape == (1, x.shape[1], 1, 1)
        x_norm = (x - mean) / t.sqrt(var + self.eps)

        # step 3. scale and shift
        x_affine = x_norm * einops.rearrange(self.weight, "c -> 1 c 1 1") + einops.rearrange(self.bias, "c -> 1 c 1 1")

        return x_affine

    def extra_repr(self):
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["num_features", "eps", "momentum"]])

class AveragePool(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        """
        return einops.reduce(x, "b c h w -> b c", "mean")

class Sequential(nn.Module):
    _modules: dict[str, nn.Module]

    def __init__(self, *modules: nn.Module | OrderedDict):
        super().__init__()
        for index, mod in enumerate(modules):
            if isinstance(mod, nn.Module):
                self._modules[str(index)] = mod
            elif isinstance(mod, OrderedDict):
                for name, submod in mod.items():
                    self._modules[name] = submod

    def __getitem__(self, index: int | str) -> nn.Module:
        if isinstance(index, int):
            index = index % len(self._modules) # deal with negative indices
            return self._modules[str(index)]
        elif isinstance(index, str):
            return self._modules[index]

    def __setitem__(self, index: int | str, module: nn.Module) -> None:
        if isinstance(index, int):
            index %= len(self._modules) # deal with negative indices
            self._modules[str(index)] = module
        elif isinstance(index, str):
            self._modules[index] = module

    def forward(self, x: Tensor) -> Tensor:
        """Chain each module together, with the output from one feeding into the next one"""
        for mod in self._modules.values():
            x = mod(x)
        return x

class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0
    ):
        """
        Same as torch.nn.Conv2d with bias=False
        Assuming kernel is square
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        sf = 1/np.sqrt(in_channels * kernel_size * kernel_size)
        weight = sf * (2 * t.rand(out_channels, in_channels, kernel_size, kernel_size) -1)
        self.weight = nn.Parameter(weight)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the functional conv2d, which you can import"""
        return conv2d(x, self.weight, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        keys = ["in_channels", "out_channels", "kernel_size", "stride", "padding"]
        return ", ".join([f"{key}={getattr(self, key)}" for key in keys])

class ReLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return t.where(x < 0, 0.0, x)

class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: int, stride: int | None = None,
                 padding: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        """Call the functional version of maxpool2d"""
        return maxpool2d(x, kernel_size=self.kernel_size, stride=self.stride,
                            padding=self.padding)

    def extra_repr(self) -> str:
        """Add additional information to the string representation of this class"""
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["kernel_size", "stride", "padding"]])

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
        wstr = f"Weight shape {self.weight.shape} min {self.weight.min():.3f}"\
                f" max: {self.weight.max():.3f} mean {self.weight.mean():.3f}. \n"
        if self.bias is not None:
            bstr = f"Bias shape {self.bias.shape} min {self.bias.min():.3f}"\
                    f" max {self.bias.max():.3f} mean {self.bias.mean():.3f}"
            wstr += bstr
        return wstr

class ResidualBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride = 1):
        """
        A single residual block with optinal downsampling
        For comptability with the pretrained model, declare the left side branch first using a 
        `Sequential`.
        
        If first_stride is > 1, this means the optional (conv + bn) should be present on the right
        branch. Declare it second using another Sequential
        """
        super().__init__()
        is_shape_preserving = (first_stride == 1) and (in_feats == out_feats) #determines if right branch is identy
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.left_side = Sequential(
            Conv2d(in_channels=in_feats, out_channels=out_feats, kernel_size=3, stride=first_stride, padding=1),
            BatchNorm2d(num_features=out_feats),
            ReLU(),
            Conv2d(in_channels=out_feats, out_channels=out_feats, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(num_features=out_feats)
        )
        if not is_shape_preserving:
            self.right_side = nn.Sequential(
                Conv2d(in_channels=in_feats, out_channels=out_feats, kernel_size=1, stride=first_stride, padding=0),
                BatchNorm2d(num_features=out_feats)
            )
        else:
            #self.right_side = None
            self.right_side = nn.Identity() # I like this one more

        self.relu = ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the forward pass. If no downsampling block is present, the addition should just add the left branch's output to the input

        x: shape (batch, in_feats, height, width)
        Return : shape (batch, out_feats, height/stride, width/stride) 
        """
        left = self.left_side(x)
        right = self.right_side(x)
        out = right + left
        out = self.relu(out)
        return out


class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        """
        An n_blocks-long sequence of ResidualBlock where only the first block uses the provided stride
        It say either n_stride > 1 or in_feats ! out_feats but I only see n_stride > 1 AND in_feats != out_feats always go together. Never separately for ResNet34, not sure which architecture it has decoupling?
        """
        super().__init__()
        self.n_blocks = n_blocks
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.first_stride = first_stride

        first_block = ResidualBlock(
            in_feats=in_feats,
            out_feats=out_feats,
            first_stride=first_stride
        )
        self.blocks = [first_block]

        for _ in range(1, n_blocks):
            self.blocks.append(ResidualBlock(
                in_feats=out_feats,
                out_feats=out_feats,
                first_stride=1
            ))

        self.blocks = Sequential(*self.blocks)


    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the forward pass.
        x: shape (batch, in_feats, height, width)
        Return: shape (batch, out_feats, height/first_stride, width / first_stride) 
        """
        x = self.blocks(x)
        return x

class ResNet34(nn.Module):
    def __init__(
            self,
            n_blocks_per_group=[3, 4, 6, 3],
            out_features_per_group=[64, 128, 256, 512],
            first_strides_per_group=[1, 2, 2, 2],
            n_classes=1000, 
            ):
        super().__init__()
        out_feats0 = 64
        self.n_blocks_per_group = n_blocks_per_group
        self.out_features_per_group = out_features_per_group
        self.first_strides_per_group = first_strides_per_group
        self.n_classes = n_classes

        # Building first conv
        self.conv1 = Sequential(
            Conv2d(in_channels = 3, out_channels=out_feats0,
                    kernel_size=7, stride=2, padding=3),
            BatchNorm2d(num_features=out_feats0),
            ReLU())

        self.max_pool1 = MaxPool2d(kernel_size=3, stride=2)

        # Building block group
        groups = []
        for nth_group in range(len(n_blocks_per_group)):
            groups.append(BlockGroup(
                n_blocks= n_blocks_per_group[nth_group],
                in_feats = out_feats0 if nth_group == 0 else out_features_per_group[nth_group-1],
                out_feats=out_features_per_group[nth_group],
                first_stride=first_strides_per_group[nth_group]
            ))

        self.groups = Sequential(*groups)

        # Building last layer
        self.avg_pool = AveragePool()
        self.linear = Linear(in_features=out_features_per_group[-1], out_features=n_classes)

    def forward(self, x: Float[Tensor, "b c h w"]) -> Float[Tensor, "b classes=1000"]:
        """
        x: shape (batch, channels, height, width)
        Return: shape (batch, n_classes)
        """
        out_conv1 = self.conv1(x)
        in_block = self.max_pool1(out_conv1)
        out_block = self.groups(in_block)
        output = self.linear(self.avg_pool(out_block))
        return output

my_resnet = ResNet34()

#%%
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models
from torchvision.transforms import v2 as transforms
from tqdm.auto import tqdm
import part2_cnns.utils as utils
import torchinfo
from dataclasses import dataclass
from fundamentals_utils import CIFAR10

from plotly_utils import line

target_resnet = models.resnet34()
utils.print_param_count(my_resnet, target_resnet)

print("My model: ", torchinfo.summary(my_resnet, input_size=(1, 3, 64, 64)), sep="\n")
print("Reference model: ", torchinfo.summary(target_resnet, input_size=(1, 3, 64, 64), depth=2), sep="\n")

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# %%
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGENET_TRANSFORM = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(t.float32, scale=True),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ]
)

### Bonus material - feature extraction
def get_resnet_for_feature_extraction(n_classes: int) -> ResNet34:
    """
    Creates a ResNet34 instance, replaces its final linear layer with a classifier for `n_claases` classes,a dn freezes all weights except the ones in this layer
    """
    # Step 1. Instantiate the model
    model = ResNet34()
    # Step 2. Load weights
    pretrained_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    mydict = model.state_dict()
    pretraineddict = pretrained_model.state_dict()
    assert len(mydict) == len(pretraineddict), "Mismatching model"

    state_load_map = {mykey : pretrainedvalue for (mykey, myvalue), (pretrainedkey, pretrainedvalue) in zip(mydict.items(), pretraineddict.items())}
    model.load_state_dict(state_dict=state_load_map)

    # Double check the value again
    for (mykey, myvalue), (pretrainedkey, pretrainedvalue) in zip(model.state_dict().items(), pretrained_model.state_dict().items()):
        assert ((myvalue - pretrainedvalue).abs() < 1e-6).all()
    print("Weights copied successfully")

    # step 3. disable gradients
    for param in model.parameters():
        param.requires_grad_(False)
    # step 4. replace the last linear with linear with `n_classes`
    model.linear = Linear(in_features=model.out_features_per_group[-1], out_features=n_classes)

    return model

@dataclass
class ResNetTrainArgs:
    batch_size: int = 64
    epochs: int = 3
    learning_rate: float = 1e-3
    n_classes: int = 10

def get_cifar(trainset_size: int = 10_000, testset_size: int = 1_000) -> tuple[Subset, Subset]:
    """Returns a subset of CIFAR-10 train and test sets (slicing the first examples)."""
    cifar_trainset = CIFAR10(train=True)
    cifar_testset = CIFAR10(train=False)

    cifar_trainset = Subset(cifar_trainset, indices=range(trainset_size))
    cifar_testset = Subset(cifar_testset, indices=range(testset_size))

    return cifar_trainset, cifar_testset

def train(args: ResNetTrainArgs) -> tuple[list[float], list[float], ResNet34]:
    """
    Performs feature extraction on ResNet, returning the model and lists of loss and accuracy
    """
    model = get_resnet_for_feature_extraction(n_classes=args.n_classes)
    model = model.to(device)

    cifar_trainset, cifar_testset = get_cifar()
    cifar_trainset = DataLoader(cifar_trainset, batch_size=args.batch_size, shuffle=True)
    cifar_testset = DataLoader(cifar_testset, batch_size=args.batch_size, shuffle=False)
    loss_list = []
    accuracy_list = []

    optimizer = t.optim.AdamW(model.linear.parameters(), lr = args.learning_rate) # I pass all and it is fine i think

    for epoch in range(args.epochs):
        pbar = tqdm(cifar_trainset)
        model.train()
        for img, label in pbar:
            # should move first then transform
            img = img.to(device)
            label = label.to(device)
            img = IMAGENET_TRANSFORM(img)

            # Forward pass
            logits = model(img) # should have shape (batch, n_classes)
            loss = nn.functional.cross_entropy(logits, label)

            # Do optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_list.append(loss.item())
            pbar.set_postfix(epoch=f"epoch={epoch}/{args.epochs}",
                             loss=f"losss={loss.item():.3f}", refresh=False)

        with t.inference_mode():
            accuracy_sum = 0
            model.eval()
            for img, label in cifar_testset:
                img = img.to(device)
                label = label.to(device)
                img = IMAGENET_TRANSFORM(img)

                # well in code, they put with t.inference_mode() only when forward
                #with t.inference_mode():
                    #logits = model(img)
                logits = model(img)
                accuracy = (t.argmax(logits, dim=-1) == label).float().sum()
                accuracy_sum += accuracy.item()
            accuracy_list.append(accuracy_sum / len(cifar_testset.dataset))

    return (loss_list, accuracy_list, model)

args = ResNetTrainArgs()
loss_list, accuracy_list, model = train(args)
# %%
line(
    y=[loss_list, [1 / args.n_classes] + accuracy_list], # start assuming a uniform accuracy of 10%
    use_secondary_yaxis=True,
    x_max=args.epochs*10_000,
    label={"x": "Num examples seen", "y1": "Cross entropy loss", "y2": "Test accuracy"},
    title="ResNet Feature Extraction",
    width=800
)
# %%
