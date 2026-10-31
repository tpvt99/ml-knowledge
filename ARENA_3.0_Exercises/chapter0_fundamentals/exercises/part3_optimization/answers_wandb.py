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
import wandb

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
from plotly_utils import bar, imshow, line

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

class ResNetFinetuner:
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
        self.testset = CIFAR10(train=False)

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

        return loss

    @t.inference_mode()
    def evaluate(self) -> float:
        """Evaluate the model on the test set and return the accuracy"""
        self.model.eval()

        pbar = tqdm(self.test_loader, desc="Evaluating")
        accuracy_sum = 0
        total_samples = 0
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            imgs = IMAGENET_TRANSFORM(imgs)

            # Output
            logits = self.model(imgs)
            accuracy = (logits.argmax(-1) == labels).sum().item()
            accuracy_sum += accuracy
            total_samples += imgs.shape[0]

            pbar.set_postfix(accuracy=f"{accuracy/total_samples:.3f}",
                             loss=f"{self.examples_seen:06}")

        accuracy = accuracy_sum / len(self.test_loader.dataset)
        self.logged_variables['accuracy'].append(accuracy)
        assert len(self.test_loader.dataset) == total_samples

        return accuracy

    def train(self) -> dict[str, list[float]]:
        self.pre_training_setup()

        for epoch in range(self.args.epochs):
            self.model.train()

            pbar = tqdm(self.train_loader, desc="Training")
            for imgs, labels in pbar:
                loss = self.training_step(imgs, labels)

                pbar.set_postfix(loss=f"{loss:.3f}",
                                 ex_seen=f"{self.examples_seen:06}",
                                 refresh=False)

            accuracy = self.evaluate()
            pbar.set_postfix(
                loss=f"{loss:.3f}",
                accuracy=f"{accuracy:.3f}",
                ex_seen=f"{self.examples_seen:06}"
            )

        return self.logged_variables

# %%
args = ResNetFinetuningArgs()
trainer = ResNetFinetuner(args)
logged_variables = trainer.train()
# %%
line(
    y=[logged_variables["loss"][: 391 * 3 + 1], [0.1] + logged_variables["accuracy"][:4]],
    x_max=len(logged_variables["loss"][: 391 * 3 + 1] * args.batch_size),
    yaxis2_range=[0, 1],
    use_secondary_yaxis=True,
    labels={"x": "Examples seen", "y1": "Cross entropy loss", "y2": "Test Accuracy"},
    title="Feature extraction with ResNet34",
    width=800,
)
#%%
@dataclass
class WandbResNetFinetuningArgs(ResNetFinetuningArgs):
    """Contains new params for use in wandb.init, as well as all the ResNetFinetuningArgs params"""
    wandb_project: str | None = "day3-resnet"
    wandb_name: str | None = None
    use_wandb: bool = False

class WandbResNetFinetuner(ResNetFinetuner):
    args: WandbResNetFinetuningArgs # adding this line helps with typechecker
    examples_seen: int = 0

    def pre_training_setup(self):
        """Initializes the wandb run using `wandb.init` and `wandb.watch`"""
        super().pre_training_setup()

        wandb.init(entity = "hehe",
                   project = self.args.wandb_project,
                   config = self.args,
                   name = self.args.wandb_name)

        wandb.watch(models = self.model, log='all', log_freq=1000)

    def training_step(
            self,
            imgs: Float[Tensor, "batch channels height width"],
            labels: Int[Tensor, "batch"],
    ) -> Float[Tensor, ""]:
        """Perform a gradient update step on a single batch of data"""

        loss = super().training_step(imgs, labels)

        # Logging wandb
        wandb.log(data=self.logged_variables, step=self.examples_seen)

        return loss


# %%
