#%%
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import torch as t
import torch
import time
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
from part3_resnets.answers_resnet import ResNet34
from plotly_utils import line, plot_train_loss_and_test_accuracy_from_trainer
from torchinfo import summary

device = t.device('cuda' if t.cuda.is_available() else 'cpu')
print(device)

MAIN = __name__ == "__main__"
# %%

# %%
# Third step is loading the model
def load_model(my_resnet, pretrained_resnet):
    mydict = my_resnet.state_dict()
    pretraineddict = pretrained_resnet.state_dict()

    state_load_dict = {
        mykey: pretrainedvalue for (mykey, myvalue), (pretrainkey, pretrainedvalue) in \
            zip(mydict.items(), pretraineddict.items())
    }

    my_resnet.load_state_dict(state_load_dict)


# %%
def get_resnet_for_feature_extraction(n_classes: int) -> ResNet34:
    '''
    Creates a ResNet34 instance, replaces its final linear layer with a classifier
    for `n_classes` classes, and freezes all weights except the ones in this layer.

    Returns the ResNet model.
    '''

    # First step is load my resnet
    my_resnet = ResNet34()
    # Second step is load pretraind resnet and assign weights into my model
    pretrained_resnet = models.resnet34(weights = models.ResNet34_Weights.IMAGENET1K_V1)
    load_model(my_resnet, pretrained_resnet)
    # Third step is frezze all weights
    for param, value in my_resnet.named_parameters():
        value.requires_grad_(False)
    # Fourth step is replace the last layer with trainable parameters
    my_resnet.linear = Linear(in_features = 512, out_features=n_classes)

    return my_resnet
#get_resnet_for_feature_extraction(10)

tests.test_get_resnet_for_feature_extraction(get_resnet_for_feature_extraction)
# %%
def get_cifar(subset: int = 1):
    IMAGE_SIZE = 224
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    IMAGENET_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Normalize(mean = IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    cifar_trainset = datasets.CIFAR10(root = './data', train=True, download=True,
                                      transform=IMAGENET_TRANSFORM)
    cifar_testset = datasets.CIFAR10(root = './data', train=False, download=True,
                                     transform=IMAGENET_TRANSFORM)
    print(f"Length of train {len(cifar_trainset)} of test: {len(cifar_testset)}") 
    if subset > 1:
        cifar_trainset = Subset(cifar_trainset, indices=range(0, len(cifar_trainset), subset))
        cifar_testset = Subset(cifar_testset, indices=range(0, len(cifar_testset), subset))
        print(f"Subset: Length of train {len(cifar_trainset)} of test: {len(cifar_testset)}") 

    return cifar_trainset, cifar_testset

@dataclass
class ResnetTrainingArgs():
    batch_size: int = 64
    epochs:int = 10
    optimizer: Type[t.optim.Optimizer] = t.optim.Adam
    learning_rate: float = 1e-3
    n_classes: int = 10
    subset: int = 1


# %%
## Full tranining code
class ResnetTrainer():
    def __init__(self, args: ResnetTrainingArgs):
        self.args = args
        self.model = get_resnet_for_feature_extraction(args.n_classes).to(device)
        self.optimizer = self.args.optimizer(self.model.parameters(),
                                            lr=args.learning_rate)
        self.trainset, self.testset = get_cifar(args.subset)

        # Store per epoch, not per timestep
        self.logged_variables = {
            "train_loss":[],
            "val_loss":[],
            "train_acc":[],
            "val_acc":[]
        }

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.args.batch_size,
                          shuffle=True)
    
    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.args.batch_size,
                          shuffle=False)
    
    def train_one_epoch(self, epoch_index: int):
        train_loss = 0
        train_acc = 0
        start = time.time()
        
        train_dataloader = self.train_dataloader()

        # Must set to train
        self.model.train()

        for i, data in enumerate(train_dataloader):
            imgs, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            self.optimizer.zero_grad()

            # Do predictions
            outputs = self.model(imgs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()

            # Do optimization
            self.optimizer.step()

            # Gathering data and report
            # Loss of batch by multiplying averaged loss by sample size
            train_loss += loss.item() * imgs.size(0)

            # Accuracy 
            preds = torch.argmax(outputs, dim=-1)
            eq = (preds == labels).float().mean()
            train_acc += eq.item() * imgs.size(0)

            # It only makes sense to print per-step loss/accuracy
            # instead of train_loss and train_acc which is not **averaged yet**
            if (i+1) % 5 == 0:
                print(f"Train: Epoch {epoch_index}|{self.args.epochs}." \
                        f" Done {100*(i+1)/len(train_dataloader):.2f}. " \
                        f" Time {(time.time() - start):.2f} elapsed"\
                        f" Per Loss: {loss.item():.2f} Per Acc {eq.item():.2f} ")
                
        # At the end store to logged
        train_loss = train_loss / len(train_dataloader.dataset)
        train_acc = train_acc / len(train_dataloader.dataset)

        self.logged_variables["train_loss"].append(train_loss)
        self.logged_variables["train_acc"].append(train_acc)

        return train_loss, train_acc

    def validation_one_epoch(self, epoch_index):
        val_loss = 0
        val_acc = 0
        start = time.time()
        test_dataloader = self.test_dataloader()

        self.model.eval()

        with torch.no_grad():

            for i, data in enumerate(test_dataloader):
                imgs, labels = data
                imgs = imgs.to(device)
                labels = labels.to(device)

                outputs = self.model(imgs)
                loss = F.cross_entropy(outputs, labels)

                # Validation loss
                val_loss += loss.item() * imgs.size(0)

                # Validation accuracy
                preds = torch.argmax(outputs, dim=-1)
                eq = (preds == labels).float().mean()
                val_acc += eq.item() * imgs.size(0)

                if (i+1) % 5 == 0:
                    print(f"Val: Epoch {epoch_index}|{self.args.epochs}." \
                        f"Done {100*(i+1)/len(test_dataloader):.2f}. " \
                        f"Time {(time.time() - start):.2f} elapsed"\
                        f" Per Loss: {loss.item():.2f} Per Acc {eq.item():.2f} ")

        # At the end store to logged
        val_loss = val_loss / len(test_dataloader.dataset)
        val_acc = val_acc / len(test_dataloader.dataset)

        self.logged_variables["val_loss"].append(val_loss)
        self.logged_variables["val_acc"].append(val_acc)

        return val_loss, val_acc

    def train(self):
        for epoch in range(self.args.epochs):
            train_loss, train_acc = self.train_one_epoch(epoch)

            val_loss, val_acc = self.validation_one_epoch(epoch)

            ## At this stage, you are now able to print loss/acc of whole dataset
            # instead of per-step like in train_one_epoch() function

            print(f"Epoch {epoch} | {self.args.epochs}."\
                  f" Train loss: {train_loss:.2f} Train acc: {train_acc:.2f}" \
                 f" Val loss: {val_loss:.2f} Val acc: {val_acc:.2f}")

            # Save model if val_loss < val_loss_min


args = ResnetTrainingArgs()
trainer = ResnetTrainer(args)
trainer.train()
# %%
