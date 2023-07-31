#%%
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import torch as t
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from einops import rearrange
import einops
from dataclasses import dataclass
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from tqdm.notebook import tqdm
from typing import List, Tuple, Dict, Type
from PIL import Image
from IPython.display import display
from pathlib import Path
import torchinfo
import json
import pandas as pd
from jaxtyping import Float, Int
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
import numpy as np

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part3_resnets"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))
os.chdir(section_dir)

from part2_cnns.solutions import get_mnist, Linear, Conv2d, Flatten, ReLU, MaxPool2d
from part3_resnets.utils import print_param_count
import part3_resnets.tests as tests
from plotly_utils import line, plot_train_loss_and_test_accuracy_from_trainer
from torchinfo import summary

device = t.device('cuda' if t.cuda.is_available() else 'cpu')
print(device)

MAIN = __name__ == "__main__"
# %%
class Sequential(nn.Module):
    _modules: Dict[str, nn.Module]

    def __init__(self, *modules: nn.Module):
        super().__init__()
        for index, mod in enumerate(modules):
            assert isinstance(mod, nn.Module)
            self._modules[str(index)] = mod

    def __getitem__(self, index: int) -> nn.Module:
        index %= len(self._modules)
        return self._modules[index]
    
    def __setitem__(self, index: int, module: nn.Module) -> None:
        index %= len(self._modules)
        self._modules[str(index)]= module

    def forward(self, x: t.Tensor)-> t.Tensor:
        for mod in self._modules.values():
            x = mod(x)

        return x

class SequentialOrderedDict(nn.Module):
    _modules: Dict[str, nn.Module]

    def __init__(self, modules: OrderedDict[str, nn.Module]):
        super().__init__()
        for key, mod in modules.items():
            self._modules[key] = mod

    def __getitem__(self, key: str) -> nn.Module:
        return self._modules[key]
    
    def __setitem__(self, key: str, module: nn.Module) -> None:
        self._modules[key]= module

    def forward(self, x: t.Tensor)-> t.Tensor:
        for mod in self._modules.values():
            x = mod(x)

        return x
    
seq = Sequential(
    nn.Linear(10,20),
    nn.ReLU(),
    nn.Softmax()
)

seq = SequentialOrderedDict(OrderedDict([
    ("linear1", nn.Linear(10, 20)),
    ("relu", nn.ReLU()),
    ("linear2", nn.Linear(20, 30))
]))
# %%
class BatchNorm2d(nn.Module):
    # Type hints below aren't functional, they're just for documentation
    running_mean: Float[Tensor, "num_features"]
    running_var: Float[Tensor, "num_features"]
    num_batches_tracked: Int[Tensor, ""] # scalar tensor

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        '''
            Like nn.BatchNorm2d with track_running_stats = True and affine = True
            Name the learnable affine parameters `weight` and `bias` in that order
        '''
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = nn.Parameter(torch.ones(size=(num_features,)))
        self.bias = nn.Parameter(torch.zeros(size=(num_features,)))

        self.register_buffer('running_mean', torch.zeros(size=(num_features,)))
        self.register_buffer('running_var', torch.ones(size=(num_features,))) # why initialized one??
        self.register_buffer('num_batches_tracked', torch.tensor(0))

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        Normalize each channel.

        Compute the variance using `torch.var(x, unbiased=False)`
        Hint: you may also find it helpful to use the argument `keepdim`.

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        '''

        if self.training:
            var = torch.var(x, dim = (0,2,3), keepdim=True, unbiased=False)
            mean = torch.mean(x, dim=(0,2,3), keepdim=True)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (1-self.momentum) * self.running_var + self.momentum * var.squeeze()
            self.num_batches_tracked += 1
        else:
            mean = einops.rearrange(self.running_mean, 'c -> 1 c 1 1')
            var = einops.rearrange(self.running_var, 'c -> 1 c 1 1')


        output = (x - mean)/torch.sqrt(var + self.eps) # (batch, channels, height, width)
        output = einops.einsum(output, self.weight, 'b c h w, c  -> b c h w') + self.bias[..., None, None]

        return output

    def extra_repr(self) -> str:
        return f"num_features {self.num_features} eps: {self.eps} momentum: {self.momentum}" \
            f"weight shape {self.weight.shape} bias shape {self.bias.shape}"

tests.test_batchnorm2d_module(BatchNorm2d)
tests.test_batchnorm2d_forward(BatchNorm2d)
tests.test_batchnorm2d_running_mean(BatchNorm2d)

# %%
# my own testing
torch_batchnorm = nn.BatchNorm2d(num_features = 10)
my_batchnorm = BatchNorm2d(num_features=10)
for i in range(3):
    inputs = torch.rand(size=(8, 10, 6, 6)) # b, c, h, w
    torch_out = torch_batchnorm(inputs)
    my_out = my_batchnorm(inputs)
    
    #assert torch.allclose(torch_batchnorm.running_var, my_batchnorm.running_var)
    assert torch.allclose(torch_batchnorm.running_mean, my_batchnorm.running_mean)
    assert torch.allclose(torch_batchnorm.num_batches_tracked, my_batchnorm.num_batches_tracked)
    assert torch.allclose(torch_batchnorm.weight, my_batchnorm.weight)
    assert torch.allclose(torch_batchnorm.bias, my_batchnorm.bias)
    assert torch.allclose(torch_out, my_out)

    print (f"Done {i}")

torch.tensor(0)
# %%
torch_out.mean()
# %%
my_out.mean()
# %%
a = torch.rand(3,2,2)
b = torch.rand(3,1,1)
einops.einsum(a, b, 'c h w, c x y -> c (h x) (w y)')

# %%
class AveragePool(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
            x: shape (batch, channels, height, width)
            return shape (batch, channels)
        '''

        # Using einops
        output = einops.reduce(x, 'b c h w -> b c' , 'mean')

        return output
# %%
class ResidualBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        '''
        A single residual block with optional downsampling.

        For compatibility with the pretrained model, declare the left side branch first using a `Sequential`.

        If first_stride is > 1, this means the optional (conv + bn) should be present on the right branch. Declare it second using another `Sequential`.
        '''
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.first_stride = first_stride

        # In the paper, residual block are 2 blocks sit inside the skip conv (no dashed), 
        # e.g. for 34-residual
        #
        # 3x3 conv, 64
        # 3x3 conv, 64
        #
        # Input has shape (b, 64, 56, 56), thus because first_stride = 1(no dashed skip connection),
        # padding must be 1 so that output side is still (b, 64, 56, 56)

        ## However, if first_stride > 1 (dashed skip connection), inputs and outputs are:
        # Input shape (b, 64, 56, 56), output shape is (b, 128, 28, 28)
        # which is doubling the channels and downsampling the height and width
        # thus in main branch, we use stride=first_strde to reduce the height and width
        # and the same time, we add skip_branch with stride=first_stride to also reduce height and width
        # we keep kernel = 3, padding = 1 in both stride=1 and stride > 1
        
        self.main_branch = Sequential(
            Conv2d(in_channels=in_feats, out_channels=out_feats, kernel_size=3, stride=first_stride,
                   padding = 1),
            BatchNorm2d(num_features=out_feats),
            ReLU(),
            Conv2d(in_channels=out_feats, out_channels=out_feats, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(num_features=out_feats)
        )

        self.skip_branch = nn.Identity()

        if first_stride > 1:
            self.skip_branch = Sequential(
                Conv2d(in_channels=in_feats, out_channels=out_feats, kernel_size=1, stride=first_stride,
                       padding=0),
                BatchNorm2d(num_features=out_feats)
            )

        self.relu = ReLU()

    def forward(self, x: Float[Tensor, "batch c h w"]) -> t.Tensor:
        '''
        Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / stride, width / stride)

        If no downsampling block is present, the addition should just add the left branch's output to the input.
        '''
        residual = self.main_branch(x)

        residual = residual + self.skip_branch(x)

        output = self.relu(residual)

        return output 
    
    def extra_repr(self) -> str:
        return f"in_feats {self.in_feats} out_feats {self.out_feats} first_stride {self.first_stride}"
 
# %%
class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride: int = 1):
        '''
            An n_blocks-long sequence of ResidualBlock where only the first block uses the provided stride.
        '''
        super().__init__()

        # In the paper we can see that the first block never has first_strde > 1
        
        ## Can use either Sequential or ModuleList
        residual_blocks = []
        residual_blocks.append(ResidualBlock(in_feats=in_feats, 
                                                 out_feats=out_feats,
                                                 first_stride=first_stride))
        
        for _ in range(1, n_blocks):
            residual_blocks.append(ResidualBlock(in_feats=out_feats,
                                                 out_feats=out_feats,
                                                 first_stride=1))

        # it blocks my view, should not use sequentials 
        self.module_sequential = Sequential(*residual_blocks)

        # Or you can do module list
        #self.module_list = torch.nn.ModuleList(residual_blocks)

        #self.blocks = residual_blocks
        #self.residual_blocks = nn.ModuleList(residual_blocks)

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        '''
        
        # If sequential:
        
        output = self.module_sequential(x)

        # If module list:
        #for i, module in enumerate(self.module_list):
            #output = module(x)
            #x = output

        return output
# %%
class ResNet34(nn.Module):
    def __init__(
            self,
            n_blocks_per_group = (3, 4, 6, 3),
            in_features_per_group = (64, 64, 128, 256),
            out_features_per_group = (64, 128, 256, 512),
            first_strides_per_group = (1, 2, 2, 2),
            n_classes = 1000
    ):
        super().__init__()
        self.n_blocks_per_group = n_blocks_per_group
        self.in_features_per_group = in_features_per_group
        self.out_features_per_group = out_features_per_group
        self.first_strides_per_group = first_strides_per_group
        self.n_classes = n_classes

        # conv1 with 7x7 kernel, 64 channels, stride=2
        self.conv1 = Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = BatchNorm2d(num_features = 64)
        self.relu1 = ReLU()

        # conv2_pool with 3x3 maxpool, stride=2
        self.conv2_maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # conv2, conv3, conv4, conv5
        blocks = []
        for i in range(len(n_blocks_per_group)):
            blocks.append(BlockGroup(n_blocks = n_blocks_per_group[i],
                                          in_feats = in_features_per_group[i],
                                          out_feats = out_features_per_group[i],
                                          first_stride = first_strides_per_group[i]))

        # This blocks view of summary 
        #self.blocks = nn.ModuleList(blocks)
        self.blocks = Sequential(*blocks)

        # average pool
        self.avg_pool = AveragePool()
        self.linear = Linear(in_features = 512, out_features=1000)


    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
            x: shape (batch, channels, height, width)
            Return: shape (batch, n_classes)
        '''
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2_maxpool(x)
        #for index, module in enumerate(self.blocks):
            #x = module(x)
        
        x = self.blocks(x)

        x = self.avg_pool(x)
        x = self.linear(x)

        return x

my_resnet = ResNet34()


# %%
# Veryfing your implementation by loading weights
def copy_weights(my_resnet : ResNet34, pretrained_resnet: models.resnet34) -> models.resnet34:
    '''
        Copy over the weights of `pretrained_resnet` to your resnet
    '''
    pass

pretrained_resnet = models.resnet34(weights = models.ResNet34_Weights)
 #%%

summary(my_resnet, input_data = torch.rand(2, 3, 224, 224), verbose=0, depth=10)
 
# %%
summary(pretrained_resnet, input_data = torch.rand(2,3,224,224))
# %%
for param in my_resnet.state_dict():
    print(param, "\t", my_resnet.state_dict()[param].size())
    print(param)

# %%
