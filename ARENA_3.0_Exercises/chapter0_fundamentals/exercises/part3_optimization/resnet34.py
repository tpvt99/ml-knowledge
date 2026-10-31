import torch as t
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
import einops
from collections import OrderedDict
import numpy as np

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
        return nn.functional.conv2d(x, self.weight, stride=self.stride, padding=self.padding)

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
        return t.nn.functional.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride,
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
        out = einops.einsum(x, self.weight, 
                            "... in_feats, out_feats in_feats -> ... out_feats")
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
