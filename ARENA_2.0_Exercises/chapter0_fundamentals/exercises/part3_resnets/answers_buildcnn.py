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

device = t.device('cuda' if t.cuda.is_available() else 'cpu')
print(device)

MAIN = __name__ == "__main__"
# %%
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(in_channels = 1, out_channels = 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = Conv2d(in_channels = 32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size = 2, stride=2, padding=0) 

        self.flatten = Flatten(start_dim = 1, end_dim=-1)
        self.fc1 = Linear(in_features = 3136, out_features = 128)
        self.relu3 = ReLU()
        self.fc2 = Linear(in_features = 128, out_features = 10)


    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        return x 
    
model = ConvNet()
print(model)
summary = torchinfo.summary(model, input_data = torch.rand(16, 1, 28, 28))
print(summary)

# %%

MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def getmnist(subset: int = 1):
    '''Return MNIST training data, sampled by the frequency given in the subset'''
    mnist_trainset = datasets.MNIST(root="./data", train=True, download=True, transform=MNIST_TRANSFORM)
    mnist_testset = datasets.MNIST(root="./data", train=False, download=True, transform=MNIST_TRANSFORM)

    if subset > 1:
        mnist_trainset = Subset(mnist_trainset, indices = range(0, len(mnist_trainset), subset))
        mnist_testset = Subset(mnist_testset, indices=range(0, len(mnist_testset), subset))

    return mnist_trainset, mnist_testset

mnist_trainset, mnist_tesetset = getmnist()
mnist_trainloader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)
mnist_testloader = DataLoader(mnist_tesetset, batch_size=64, shuffle=False)
# %%
model = ConvNet().to(device)

batch_size = 64
epochs = 3

mnist_trainset, _ = getmnist(subset=10)
mnist_trainloader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)

optimizer = torch.optim.Adam(model.parameters())
loss_list = []

for epoch in tqdm(range(epochs)):
    for imgs, labels in mnist_trainloader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())


line(
    loss_list,
    yaxis_range=[0, max(loss_list) + 0.1],
    labels={"x": "Num batches seen", "y": "Cross entropy loss"},
    title="ConvNet training on MNIST",
    width=700
)
# %%
@dataclass
class ConvNetTrainingArgs():
    '''
    Defining this class implicitly creates an __init__ method, which sets arguments as 
    given below, e.g. self.batch_size = 64. Any of these arguments can also be overridden
    when you create an instance, e.g. args = ConvNetTrainingArgs(batch_size=128).
    '''
    batch_size: int = 64
    epochs: int = 3
    optimizer: Type[t.optim.Optimizer] = t.optim.Adam
    learning_rate: float = 1e-3
    subset: int = 10

class ConvNetTrainer:
    def __init__(self, args: ConvNetTrainingArgs):
        self.args = args
        self.model = ConvNet().to(device)
        self.optimizer = args.optimizer(self.model.parameters(), lr = args.learning_rate)
        self.trainset, self.testset = getmnist(subset = args.subset)
        self.logged_variables = {"train_loss": [], "val_loss":[], "val_acc_epoch":[], "val_acc":[]}

    def training_step(self, imgs: Tensor, labels: Tensor):
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = self.model(imgs)
        loss = F.cross_entropy(logits, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.logged_variables["train_loss"].append(loss.item())

        return loss
    
    def validation_step(self, imgs: Tensor, labels: Tensor):
        with torch.inference_mode():
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = self.model(imgs)
            loss = F.cross_entropy(logits, labels)

            pred_labels = torch.argmax(logits, dim=-1)
            acc = (labels == pred_labels).cpu().float().mean().item()

            self.logged_variables["val_loss"].append(loss.item())
            self.logged_variables["val_acc_epoch"].append(acc)

            return loss.item(), acc

    def train_dataloader(self):
        return DataLoader(dataset = self.trainset, batch_size=self.args.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset = self.testset, batch_size = self.args.batch_size, shuffle=False)

    def train(self):
        progress_bar = tqdm(total = self.args.epochs * len(self.trainset) // self.args.batch_size)
        for epoch in range(self.args.epochs):
            for imgs, labels in self.train_dataloader():
                loss = self.training_step(imgs, labels)
                desc = f"Epoch {epoch+1}/{self.args.epochs}, Loss: {loss.item():.2f} {len(self.trainset)}"
                progress_bar.set_description(desc)
                progress_bar.update()

            self.val() 
            self.logged_variables["val_acc"].append(np.mean(self.logged_variables["val_acc_epoch"]))
            self.logged_variables["val_acc_epoch"] = [] 

    def val(self):
        progress_bar = tqdm(total = len(self.testset) // self.args.batch_size)
        for imgs, labels in self.val_dataloader():
            loss, acc = self.validation_step(imgs, labels)
            desc = f"Loss: {loss:.2f} and Accuracy {acc:.2f}"
            progress_bar.set_description(desc)
            progress_bar.update()

args = ConvNetTrainingArgs(batch_size=128)
trainer = ConvNetTrainer(args)
trainer.train()
trainer.val()
                    
line(
    trainer.logged_variables["train_loss"], 
    yaxis_range=[0, max(trainer.logged_variables["train_loss"]) + 0.1],
    labels={"x": "Num batches seen", "y": "Cross entropy loss"}, 
    title="ConvNet training on MNIST",
    width=700
)

line(
    trainer.logged_variables["val_acc"], 
    yaxis_range=[0, max(trainer.logged_variables["val_acc"]) + 0.1],
    labels={"x": "Epochs", "y": "Validation accuracy"}, 
    title="ConvNet training on MNIST",
    width=700
)





# %%
