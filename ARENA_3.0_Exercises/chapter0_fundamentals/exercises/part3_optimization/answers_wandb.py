#%%
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models
from torchvision.transforms import v2 as transforms
from tqdm.auto import tqdm
import torchinfo
from dataclasses import dataclass
from pathlib import Path
import sys
import torch as t
from jaxtyping import Float, Int
from torch import Tensor

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part3_optimization"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
print(exercises_dir)
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

# Then we can import these 2 files
from fundamentals_utils import CIFAR10, IMAGENET_TRANSFORM
from part3_optimization.resnet34 import ResNet34, Linear

#%%
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
    # step 3. disable gradients
    for param in model.parameters():
        param.requires_grad_(False)
    # step 4. replace the last linear with linear with `n_classes`
    model.linear = Linear(in_features=model.out_features_per_group[-1], out_features=n_classes)

    return model

get_resnet_for_feature_extraction(10)
# %%
device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

@dataclass
class ResNetFinetuningArgs:
    n_classes: int = 10
    batch_size: int = 128
    epochs: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 0.0

class ResNetFintuner:
    def __init__(self, args: ResNetFinetuningArgs):
        self.args = args

    def pre_training_setup(self):
        self.model = get_resnet_for_feature_extraction(self.args.n_classes).to(device)
        self.optimizer = t.optim.AdamW(
            params=self.model.linear.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )

        self.trainset = CIFAR10(train=True)
        self.testset = CIFAR10(test=True)

        self.train_loader = DataLoader(self.trainset, 
                                       batch_size=self.args.batch_size,
                                       shuffle=True)

        self.test_loader = DataLoader(self.testset,
                                      batch_size=self.args.batch_size,
                                      shuffle=False)

        self.logged_variables = {'loss': [], 'accuracy': []}
        self.examples_seen = 0

    def training_step(
            self,
            imgs: Float[Tensor, "batch channels height width"],
            labels: Int[Tensor, "batch"],
    ) -> Float[Tensor, ""]:
        """Perform a gradient update step on a single batch of data"""

        # CPU -> GPU first then perform preprocessing on GPU
        imgs = imgs.to(device)
        labels = labels.to(device)
        imgs = IMAGENET_TRANSFORM(imgs)

        # Find output
        logits = self.model(imgs)
        # Loss
        loss = t.nn.functional.cross_entropy(logits, labels)
        # Do optimization
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Logging
        self.logged_variables['loss'].append(loss.item())
        self.examples_seen += imgs.shape[0]

    @t.inference_mode()
    def evaluate(self) -> float:
        """Evaluate the model on the test set and return the accuracy"""
        self.model.eval()

        pbar = tqdm(self.test_loader)
        accuracy_sum = 0
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            imgs = IMAGENET_TRANSFORM(imgs)

            # Output
            logits = self.model(imgs)
            accuracy = (logits.argmax(-1) == labels).sum().item()
            accuracy_sum += accuracy

            pbar.set_postfix({})

        self.logged_variables['accuracy'].append(accuracy_sum / len(self.test_loader.dataset))

# %%
x = t.rand(10,100,100)
print(len(x))
# %%
testset = CIFAR10(train=True)

test_loader = DataLoader(testset,
                                batch_size=256,
                                shuffle=False)
#%%
pbar = tqdm(test_loader, "processing ")
count = 1
for item in pbar:
    pbar.set_postfix({"count": count})
    count+=1
# %%
