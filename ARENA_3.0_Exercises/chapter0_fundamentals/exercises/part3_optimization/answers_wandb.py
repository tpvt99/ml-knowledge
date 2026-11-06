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
from part3_optimization import tests

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
    epochs: int = 3
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
    examples_seen: int # this should be separate from self.examples_seen as it should be step for wandb

    def pre_training_setup(self):
        """Initializes the wandb run using `wandb.init` and `wandb.watch`"""
        super().pre_training_setup()

        wandb.init(entity = "tpvt99", # entity seem must be your name
                   project = self.args.wandb_project,
                   config = self.args,
                   name = self.args.wandb_name,
                   mode='online' if self.args.use_wandb else 'disabled')

        wandb.watch(models = self.model, log='all', log_freq=50)

    def training_step(
            self,
            imgs: Float[Tensor, "batch channels height width"],
            labels: Int[Tensor, "batch"],
    ) -> Float[Tensor, ""]:
        """Perform a gradient update step on a single batch of data"""

        loss = super().training_step(imgs, labels)

        # Logging wandb
        examples_seen = self.examples_seen
        wandb.log(data={'loss': loss}, step=examples_seen)

        return loss

    @t.inference_mode()
    def evaluate(self) -> float:
        "Equivalent to ResNetFinetuner.evaluate, but logging the accuracy to wandb"
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

            pbar.set_postfix(accuracy=f"{accuracy_sum/total_samples:.3f}",
                             ex_seen=f"{self.examples_seen:06}")

        accuracy = accuracy_sum / len(self.test_loader.dataset)
        self.logged_variables['accuracy'].append(accuracy)

        wandb.log({'accuracy': accuracy}, step = self.examples_seen)

        return accuracy

    def train(self) -> None:
        """Equivalent to ResNetFintuner.train but with wandb intergration"""
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
            #pbar.set_postfix(
                #loss=f"{loss:.3f}",
                #accuracy=f"{accuracy:.3f}",
                #ex_seen=f"{self.examples_seen:06}"
            #)
        wandb.finish()

        return self.logged_variables

# %%
args = WandbResNetFinetuningArgs(use_wandb=False)
trainer = WandbResNetFinetuner(args)
trainer.train()
# %%
sweep_config = dict(
    method='random',
    metric= {'name': 'accuracy', 'goal': 'maximize'},
    parameters=dict(
        learning_rate = dict(min=1e-4, max=1e-1, distribution='log_uniform_values'),
        batch_size = dict(values = [32, 64, 128, 256]),
        use_weight_decay = dict(values = [True, False], probabilities = [0.5, 0.5]),
        weight_decay = dict(min=1e-4, max=1e-2, distribution='log_uniform_values')
    )
)

def update_args(args: WandbResNetFinetuningArgs, sampled_parameters: dict) -> WandbResNetFinetuningArgs:
    """
    Returns a new args object with modified values. The dictionary `sampled_parameters` will have
    the same keys as your `sweep_config["parameters"]` dict and values equal to the sampled
    values of those hyperparameters
    """
    assert set(sampled_parameters.keys()) == set(sweep_config["parameters"].keys())

    use_weight_decay = sampled_parameters['use_weight_decay']
    for key, value in sampled_parameters.items():
        if hasattr(args, key):
            setattr(args, key, value)

            if key == 'weight_decay' and (not use_weight_decay):
                setattr(args, key, 0)

    return args

tests.test_sweep_config(sweep_config)
tests.test_update_args(update_args, sweep_config)
# %%
def train():
    # Define args and initialize wandb
    args = WandbResNetFinetuningArgs(use_wandb=True)
    # We put reinit = False here so that in .train() dont need redo .init()
    wandb.init(project=args.wandb_project, name=args.wandb_name, reinit=False)

    # After initializing wandb, we can update args using `wandb.config`
    args = update_args(args, dict(wandb.config))

    # Train the model with these new hyperparameters (the second wandb.init will be ignored)
    trainer = WandbResNetFinetuner(args)
    trainer.train()

try:
    # If you ignore project, it creates new project. it somehow does not care the .init in train() which i already add the project name
    sweep_id = wandb.sweep(sweep=sweep_config, project='day3-resnet')
    wandb.agent(sweep_id=sweep_id, function=train, count=3)
finally:
    wandb.teardown()
    wandb.finish()

# %%
