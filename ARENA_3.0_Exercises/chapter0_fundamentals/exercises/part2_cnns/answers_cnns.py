#%%
import json
import sys
from collections import namedtuple, OrderedDict
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
        wstr = f"Weight shape {self.weight.shape} min {self.weight.min():.3f}"\
                f" max: {self.weight.max():.3f} mean {self.weight.mean():.3f}. \n"
        if self.bias is not None:
            bstr = f"Bias shape {self.bias.shape} min {self.bias.min():.3f}"\
                    f" max {self.bias.max():.3f} mean {self.bias.mean():.3f}"
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

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = Flatten(start_dim=1, end_dim=-1)

        # Layer 1, I will do it raw here
        lay1_in_features = 28 * 28
        lay1_out_features = 100
        #sf = 1 / np.sqrt(lay1_in_features)
        #weight1 = 2 * t.rand(lay1_out_features, lay1_in_features) - 1 # make [-1, 1]
        #weight1 = sf * weight1 # make [-sf, +sf]
        #self.weight1 = nn.Parameter(weight1)

        #bias1 = 2 * t.rand(lay1_out_features) - 1 # make [-1, 1]
        #bias1 = sf * bias1 # make [-sf, +sf]
        #self.bias1 = nn.Parameter(bias1)
        self.linear1 = Linear(lay1_in_features, lay1_out_features, True)

        # Layer 2, I will do it raw here
        lay2_in_features = lay1_out_features
        lay2_out_features = 10
        #sf = 1 / np.sqrt(lay2_in_features)
        #weight2 = sf * (2 * t.rand(lay2_out_features, lay2_in_features) -1)
        #self.weight2 = nn.Parameter(weight2)

        #bias2 = sf * (2 * t.rand(lay2_out_features) - 1)
        #self.bias2 = nn.Parameter(bias2)
        self.linear2 = Linear(lay2_in_features, lay2_out_features, True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.flatten(x)
        linear1 = self.linear1(x)
        relu1 = ReLU()(linear1)
        out = self.linear2(relu1)
        return out


tests.test_mlp_module(SimpleMLP)
tests.test_mlp_forward(SimpleMLP)
# %%
MNIST_TRANSFORM = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(t.float32, scale=True),
        transforms.Normalize([0.1307], [0.3081]),
    ]
)

def get_mnist(trainset_size: int = 10_000, testset_size: int = 1000) -> tuple[Subset, Subset]:
    """Return a subset of MNIST training data"""

    #Get original datasets, which are downloaded to "./data" for future use
    mnist_trainset = datasets.MNIST(exercises_dir / "data", train = True, download=True, transform = MNIST_TRANSFORM)
    mnist_testset = datasets.MNIST(exercises_dir / "data", train=False, download=True, transform=MNIST_TRANSFORM)

    # Return a subset of the original datasets
    mnist_trainset = Subset(mnist_trainset, indices=range(trainset_size))
    mnist_testset = Subset(mnist_testset, indices=range(testset_size))

    return mnist_trainset, mnist_testset

mnist_trainset, mnist_testset = get_mnist()
mnist_trainloader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)
mnist_testloader = DataLoader(mnist_testset, batch_size=64, shuffle=False)

# Get the first batch of test data, by starting to iterate over `mnist_testloader`
for img_batch, label_batch in mnist_testloader:
    print(f"{img_batch.shape=}\n{label_batch.shape=}\n")
    break

for img, label in mnist_testset:
    print(f"{img.shape=}\b{label=}")
    break

t.testing.assert_close(img, img_batch[0])
assert label == label_batch[0].item()

# %%
device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
print(device)

model = SimpleMLP().to(device)

batch_size = 128
epochs = 3

mnist_trainset, _ = get_mnist()
mnist_trainloader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)

optimizer = t.optim.Adam(model.parameters(), lr=1e-3)
loss_list = []

for epoch in range(epochs):
    pbar = tqdm(mnist_trainloader)

    for imgs, labels in pbar:
        # Move data to device, perform forward pass
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)

        # Calculate loss, perform backward pass
        loss = F.cross_entropy(logits, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Update logs and progress bar
        loss_list.append(loss.item())
        pbar.set_postfix(epoch=f"{epoch+1}/{epochs}", loss=f"{loss:.3f}", refresh=False)

line(
    loss_list,
    x_max = epochs*len(mnist_trainset),
    labels={"x" : "Examples seen", "y": "Cross entropy loss"},
    title = "SimpleMLP training on MNIST",
    width=700
) 

# %%
@dataclass
class SimpleMLPTrainingArgs:
    """
    Defining this class implicitly creates an __init__ method, which sets arguments as below, e.g.
    self.batch_size=64. Any of these fields can also be overridden when you create an instance, e.g.
    SimpleMLPTrainingArgs(batch_size=128)
    """

    batch_size: int = 64
    epochs: int = 3
    learning_rate: float = 1e-3

def train(args: SimpleMLPTrainingArgs) -> tuple[list[float], SimpleMLP]:
    """
    Train and returns the model, using training parameters from `args` object. Returns the model,
    and the loss list 
    """

    model = SimpleMLP().to(device)

    mnist_trainset, mnist_testset = get_mnist()
    mnist_trainloader = DataLoader(mnist_trainset, batch_size=args.batch_size, shuffle=True)
    mnist_testloader = DataLoader(mnist_testset, batch_size=args.batch_size, shuffle=False)

    optimizer = t.optim.Adam(model.parameters(), lr = args.learning_rate)
    loss_list = []
    accuracy_list = []

    for epoch in range(args.epochs):
        pbar = tqdm(mnist_trainloader)

        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)

            # Calculate loss
            loss = F.cross_entropy(logits, labels)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # Update logs and progress bar
            loss_list.append(loss.item())
            pbar.set_postfix(epoch=f"epoch={epoch}/{args.epochs}",
                             loss=f"losss={loss.item():.3f}", refresh=False)

        with t.inference_mode():
            sum_accuracy = 0
            for imgs, labels in mnist_testloader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                prob = t.softmax(logits, dim=-1) # no need softmax
                accuracy = (t.argmax(prob, dim=-1) == labels).float().mean()
                sum_accuracy += accuracy.item()
            accuracy_list.append(np.array(sum_accuracy).mean())

    return loss_list, accuracy_list, model

args = SimpleMLPTrainingArgs()
loss_list, accuracy_list, model = train(args)

line(
    y = [loss_list, [0.1] + accuracy_list],
    use_secondary_yaxis=True,
    x_max = epochs*len(mnist_trainset),
    labels={"x" : "Examples seen", "y": "Cross entropy loss"},
    title = "SimpleMLP training on MNIST",
    width=700
)
# %%
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
        return t.nn.functional.conv2d(x, self.weight, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        keys = ["in_channels", "out_channels", "kernel_size", "stride", "padding"]
        return ", ".join([f"{key}={getattr(self, key)}" for key in keys])

tests.test_conv2d_module(Conv2d) 
m = Conv2d(in_channels=24, out_channels=12, kernel_size=3, stride=2, padding=1)
print(f"Manually verify that this is an informative repr: {m}")

class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: int, stride: int | None = None,
                 padding: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        """Call the functional version of maxpool2d"""
        return F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride,
                            padding=self.padding)

    def extra_repr(self) -> str:
        """Add additional information to the string representation of this class"""
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["kernel_size", "stride", "padding"]])

m = MaxPool2d(10, 1, 3)
print(m)
        
# %%
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

#x = t.rand(3,10)
#temp = Sequential(Sequential(Linear(10, 20), ReLU(), Linear(20,20)),
                #ReLU(),
                #Linear(20,10))
#print(temp)
#temp(x)

#from collections import OrderedDict

#temp = Sequential(
    #OrderedDict([("linear1", Linear(10, 20)), 
                 #("relu1", ReLU()), 
                 #("linear2", Linear(20,20))]),
                #ReLU(),
                #Linear(20,10))
#print(temp)
#temp(x)

# %%
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


tests.test_batchnorm2d_module(BatchNorm2d)
tests.test_batchnorm2d_forward(BatchNorm2d)
tests.test_batchnorm2d_running_mean(BatchNorm2d)
tests.test_batchnorm2d_running_stats_detached(BatchNorm2d)

class AveragePool(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        """
        return einops.reduce(x, "b c h w -> b c", "mean")

tests.test_averagepool(AveragePool)

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

tests.test_residual_block(ResidualBlock)

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

        #self.blocks = nn.ModuleList(self.blocks)
        self.blocks = Sequential(*self.blocks)

        # An issue when i try. If using Sequential, then need *self.blocks to extract to *modules in Sequential we did. Then in forward, just self.blocks(x) without for loop. 2/ If you put self.blocks = [], torch won't track parameters even in forward you do for block in self.blocks. You have to use nn.ModuleList or nn.Sequential which supports list. If we use our own Sequential, then just dont for loop in forward

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the forward pass.
        x: shape (batch, in_feats, height, width)
        Return: shape (batch, out_feats, height/first_stride, width / first_stride) 
        """
        #for block in self.blocks:
            #x = block(x)
        x = self.blocks(x)
        return x

tests.test_block_group(BlockGroup)

#%%
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
utils.print_param_count(my_resnet)

# %%
# Test 1 via helper function `print_param_count`
target_resnet = models.resnet34()
utils.print_param_count(my_resnet, target_resnet)
# %%
# Test 2 via torchinfo.summary
print("My model: ", torchinfo.summary(my_resnet, input_size=(1, 3, 64, 64)), sep="\n")
print("Reference model: ", torchinfo.summary(target_resnet, input_size=(1, 3, 64, 64), depth=2), sep="\n")
# %%
def copy_weights(my_resnet: ResNet34, pretrained_resnet: models.resnet.ResNet) -> ResNet34:
    """
    Copy weights of `pretrained_resnet` to your resnet
    """
    # Get the state dictionaries for each model, check they have the same number of parameters and buffers
    mydict = my_resnet.state_dict()
    pretraineddict = pretrained_resnet.state_dict()
    assert len(mydict) == len(pretraineddict), "Mismatching state dictionaries."

    # Define a dictionary mapping the names of your parameters/buffers to their values in the pretrained model
    state_dict_to_load = {
        mykey: pretrainedvalue for (mykey, myvalue), (pretrainedkey, pretrainedvalue) in zip(mydict.items(), pretraineddict.items())
    }

    # Load in this dictionary to your model
    my_resnet.load_state_dict(state_dict_to_load)

    return my_resnet

pretrained_resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1).to(device)
my_resnet = copy_weights(my_resnet, pretrained_resnet).to(device)
for (mykey, myvalue), (pretrainedkey, pretrainedvalue) in zip(my_resnet.state_dict().items(), pretrained_resnet.state_dict().items()):
    assert ((myvalue - pretrainedvalue).abs() < 1e-6).all()
print("Weights copied successfully")
# %%
# Running the model
IMAGE_FILENAMES = [
    "chimpanzee.jpg",
    "golden_retriever.jpg",
    "platypus.jpg",
    "frogs.jpg",
    "fireworks.jpg",
    "astronaut.jpg",
    "iguana.jpg",
    "volcano.jpg",
    "goofy.jpg",
    "dragonfly.jpg",
]

IMAGE_FOLDER = section_dir / "resnet_inputs"
images = [Image.open(IMAGE_FOLDER / filename) for filename in IMAGE_FILENAMES]
display(images[0])
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

prepared_images = t.stack([IMAGENET_TRANSFORM(img) for img in images], dim = 0).to(device)
assert prepared_images.shape == (len(images), 3, IMAGE_SIZE, IMAGE_SIZE)
# %%
