# %%

import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import pandas as pd
import torch as t
from torch import Tensor, optim
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from typing import Callable, Iterable, Tuple, Optional, Type
from jaxtyping import Float
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from dataclasses import dataclass
from tqdm.notebook import tqdm
from pathlib import Path
import numpy as np
from IPython.display import display, HTML

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part4_optimization"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))
os.chdir(section_dir)

from plotly_utils import bar, imshow, plot_train_loss_and_test_accuracy_from_trainer
from part3_resnets.solutions import IMAGENET_TRANSFORM, ResNet34, get_resnet_for_feature_extraction
from part4_optimization.utils import plot_fn, plot_fn_with_points
import part4_optimization.tests as tests

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"

# %% 1️⃣ OPTIMIZERS

def pathological_curve_loss(x: t.Tensor, y: t.Tensor):
	# Example of a pathological curvature. There are many more possible, feel free to experiment here!
	x_loss = t.tanh(x) ** 2 + 0.01 * t.abs(x)
	y_loss = t.sigmoid(y)
	return x_loss + y_loss


if MAIN:
	plot_fn(pathological_curve_loss)

# %%

def opt_fn_with_sgd(fn: Callable, xy: t.Tensor, lr=0.001, momentum=0.98, n_iters: int = 100):
	'''
	Optimize the a given function starting from the specified point.

	xy: shape (2,). The (x, y) starting point.
	n_iters: number of steps.
	lr, momentum: parameters passed to the torch.optim.SGD optimizer.

	Return: (n_iters, 2). The (x,y) BEFORE each step. So out[0] is the starting point.
	'''
	assert xy.requires_grad

	xys = t.zeros((n_iters, 2))

	# YOUR CODE HERE: run optimization, and populate `xys` with the coordinates before each step
	optimizer = optim.SGD([xy], lr=lr, momentum=momentum)

	for i in range(n_iters):
		xys[i] = xy.detach()
		out = fn(xy[0], xy[1])
		out.backward()
		optimizer.step()
		optimizer.zero_grad()
		
	return xys

# %%


if MAIN:
	points = []
	
	optimizer_list = [
		(optim.SGD, {"lr": 0.1, "momentum": 0.0}),
		(optim.SGD, {"lr": 0.02, "momentum": 0.99}),
	]
	
	for optimizer_class, params in optimizer_list:
		xy = t.tensor([2.5, 2.5], requires_grad=True)
		xys = opt_fn_with_sgd(pathological_curve_loss, xy=xy, lr=params['lr'], momentum=params['momentum'])
	
		points.append((xys, optimizer_class, params))
	
	plot_fn_with_points(pathological_curve_loss, points=points)

# %%

class SGD:
	def __init__(
		self, 
		params: Iterable[t.nn.parameter.Parameter], 
		lr: float, 
		momentum: float = 0.0, 
		weight_decay: float = 0.0
	):
		'''Implements SGD with momentum.

		Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
			https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD

		'''
		params = list(params) # turn params into a list (because it might be a generator)
		self.params = params
		self.lr = lr
		self.mu = momentum
		self.lmda = weight_decay
		self.t = 0

		self.gs = [t.zeros_like(p) for p in self.params]

	def zero_grad(self) -> None:
		for param in self.params:
			param.grad = None

	@t.inference_mode()
	def step(self) -> None:
		for i, (g, param) in enumerate(zip(self.gs, self.params)):
			# Implement the algorithm from the pseudocode to get new values of params and g
			new_g = param.grad
			if self.lmda != 0:
				new_g = new_g + (self.lmda * param)
			if self.mu != 0 and self.t > 0:
				new_g = (self.mu * g) + new_g
			# Update params (remember, this must be inplace)
			self.params[i] -= self.lr * new_g
			# Update g
			self.gs[i] = new_g
		self.t += 1

	def __repr__(self) -> str:
		return f"SGD(lr={self.lr}, momentum={self.mu}, weight_decay={self.lmda})"



if MAIN:
	tests.test_sgd(SGD)

# %%

class RMSprop:
	def __init__(
		self,
		params: Iterable[t.nn.parameter.Parameter],
		lr: float = 0.01,
		alpha: float = 0.99,
		eps: float = 1e-08,
		weight_decay: float = 0.0,
		momentum: float = 0.0,
	):
		'''Implements RMSprop.

		Like the PyTorch version, but assumes centered=False
			https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html

		'''
		params = list(params) # turn params into a list (because it might be a generator)
		self.params = params
		self.lr = lr
		self.eps = eps
		self.mu = momentum
		self.lmda = weight_decay
		self.alpha = alpha

		self.bs = [t.zeros_like(p) for p in self.params]
		self.vs = [t.zeros_like(p) for p in self.params]

	def zero_grad(self) -> None:
		for p in self.params:
			p.grad = None

	@t.inference_mode()
	def step(self) -> None:
		for i, (p, b, v) in enumerate(zip(self.params, self.bs, self.vs)):
			new_g = p.grad
			if self.lmda != 0:
				new_g = new_g + self.lmda * p
			new_v = self.alpha * v + (1 - self.alpha) * new_g.pow(2)
			self.vs[i] = new_v
			if self.mu > 0:
				new_b = self.mu * b + new_g / (new_v.sqrt() + self.eps)
				p -= self.lr * new_b
				self.bs[i] = new_b
			else:
				p -= self.lr * new_g / (new_v.sqrt() + self.eps)

	def __repr__(self) -> str:
		return f"RMSprop(lr={self.lr}, eps={self.eps}, momentum={self.mu}, weight_decay={self.lmda}, alpha={self.alpha})"



if MAIN:
	tests.test_rmsprop(RMSprop)

# %%

class Adam:
	def __init__(
		self,
		params: Iterable[t.nn.parameter.Parameter],
		lr: float = 0.001,
		betas: Tuple[float, float] = (0.9, 0.999),
		eps: float = 1e-08,
		weight_decay: float = 0.0,
	):
		'''Implements Adam.

		Like the PyTorch version, but assumes amsgrad=False and maximize=False
			https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
		'''
		params = list(params) # turn params into a list (because it might be a generator)
		self.params = params
		self.lr = lr
		self.beta1, self.beta2 = betas
		self.eps = eps
		self.lmda = weight_decay
		self.t = 1

		self.gs = [t.zeros_like(p) for p in self.params]
		self.ms = [t.zeros_like(p) for p in self.params]
		self.vs = [t.zeros_like(p) for p in self.params]

	def zero_grad(self) -> None:
		for p in self.params:
			p.grad = None

	@t.inference_mode()
	def step(self) -> None:
		for i, (p, g, m, v) in enumerate(zip(self.params, self.gs, self.ms, self.vs)):
			new_g = p.grad
			if self.lmda != 0:
				new_g = new_g + self.lmda * p
			self.gs[i] = new_g
			new_m = self.beta1 * m + (1 - self.beta1) * new_g
			new_v = self.beta2 * v + (1 - self.beta2) * new_g.pow(2)
			self.ms[i] = new_m
			self.vs[i] = new_v
			m_hat = new_m / (1 - self.beta1 ** self.t)
			v_hat = new_v / (1 - self.beta2 ** self.t)
			p -= self.lr * m_hat / (v_hat.sqrt() + self.eps)
		self.t += 1

	def __repr__(self) -> str:
		return f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}, weight_decay={self.lmda})"



if MAIN:
	tests.test_adam(Adam)

# %%

class AdamW:
	def __init__(
		self,
		params: Iterable[t.nn.parameter.Parameter],
		lr: float = 0.001,
		betas: Tuple[float, float] = (0.9, 0.999),
		eps: float = 1e-08,
		weight_decay: float = 0.0,
	):
		'''Implements Adam.

		Like the PyTorch version, but assumes amsgrad=False and maximize=False
			https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
		'''
		params = list(params) # turn params into a list (because it might be a generator)
		self.params = params
		self.lr = lr
		self.beta1, self.beta2 = betas
		self.eps = eps
		self.lmda = weight_decay
		self.t = 1

		self.gs = [t.zeros_like(p) for p in self.params]
		self.ms = [t.zeros_like(p) for p in self.params]
		self.vs = [t.zeros_like(p) for p in self.params]

	def zero_grad(self) -> None:
		for p in self.params:
			p.grad = None

	@t.inference_mode()
	def step(self) -> None:
		for i, (p, g, m, v) in enumerate(zip(self.params, self.gs, self.ms, self.vs)):
			new_g = p.grad
			if self.lmda != 0:
				# new_g = new_g + self.lmda * p
				p -= p * self.lmda * self.lr
			self.gs[i] = new_g
			new_m = self.beta1 * m + (1 - self.beta1) * new_g
			new_v = self.beta2 * v + (1 - self.beta2) * new_g.pow(2)
			self.ms[i] = new_m
			self.vs[i] = new_v
			m_hat = new_m / (1 - self.beta1 ** self.t)
			v_hat = new_v / (1 - self.beta2 ** self.t)
			p -= self.lr * m_hat / (v_hat.sqrt() + self.eps)
		self.t += 1

	def __repr__(self) -> str:
		return f"AdamW(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}, weight_decay={self.lmda})"



if MAIN:
	tests.test_adamw(AdamW)

# %%

def opt_fn(fn: Callable, xy: t.Tensor, optimizer_class, optimizer_hyperparams: dict, n_iters: int = 100):
	'''Optimize the a given function starting from the specified point.

	optimizer_class: one of the optimizers you've defined, either SGD, RMSprop, or Adam
	optimzer_kwargs: keyword arguments passed to your optimiser (e.g. lr and weight_decay)
	'''
	assert xy.requires_grad

	xys = t.zeros((n_iters, 2))
	optimizer = optimizer_class([xy], **optimizer_hyperparams)

	for i in range(n_iters):
		xys[i] = xy.detach()
		out = fn(xy[0], xy[1])
		out.backward()
		optimizer.step()
		optimizer.zero_grad()
	
	return xys

# %%


if MAIN:
	points = []
	
	optimizer_list = [
		(SGD, {"lr": 0.03, "momentum": 0.99}),
		(RMSprop, {"lr": 0.02, "alpha": 0.99, "momentum": 0.8}),
		(Adam, {"lr": 0.2, "betas": (0.99, 0.99), "weight_decay": 0.005}),
	]
	
	for optimizer_class, params in optimizer_list:
		xy = t.tensor([2.5, 2.5], requires_grad=True)
		xys = opt_fn(pathological_curve_loss, xy=xy, optimizer_class=optimizer_class, optimizer_hyperparams=params)
		points.append((xys, optimizer_class, params))
	
	plot_fn_with_points(pathological_curve_loss, points=points)

# %%

def bivariate_gaussian(x, y, x_mean=0.0, y_mean=0.0, x_sig=1.0, y_sig=1.0):
	norm = 1 / (2 * np.pi * x_sig * y_sig)
	x_exp = (-1 * (x - x_mean) ** 2) / (2 * x_sig**2)
	y_exp = (-1 * (y - y_mean) ** 2) / (2 * y_sig**2)
	return norm * t.exp(x_exp + y_exp)

def neg_trimodal_func(x, y):
	z = -bivariate_gaussian(x, y, x_mean=1.0, y_mean=-0.5, x_sig=0.2, y_sig=0.2)
	z -= bivariate_gaussian(x, y, x_mean=-1.0, y_mean=0.5, x_sig=0.2, y_sig=0.2)
	z -= bivariate_gaussian(x, y, x_mean=-0.5, y_mean=-0.8, x_sig=0.2, y_sig=0.2)
	return z


if MAIN:
	plot_fn(neg_trimodal_func, x_range=(-2, 2), y_range=(-2, 2))

# %%

def rosenbrocks_banana_func(x: t.Tensor, y: t.Tensor, a=1, b=100) -> t.Tensor:
	return (a - x) ** 2 + b * (y - x**2) ** 2 + 1


if MAIN:
	plot_fn(rosenbrocks_banana_func, x_range=(-2, 2), y_range=(-1, 3), log_scale=True)

# %%

class SGD:

	def __init__(self, params, **kwargs):
		'''Implements SGD with momentum.

		Accepts parameters in groups, or an iterable.

		Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
			https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
		'''

		if not isinstance(params, (list, tuple)):
			params = [{"params": params}]

		# assuming params is a list of dictionaries, we make self.params also a list of dictionaries (with other kwargs filled in)
		default_param_values = dict(momentum=0.0, weight_decay=0.0)

		# creating a list of param groups, which we'll iterate over during the step function
		self.param_groups = []
		# creating a list of params, which we'll use to check whether a param has been added twice
		params_to_check_for_duplicates = set()

		for param_group in params:
			# update param_group with kwargs passed in init; if this fails then update with the default values
			param_group = {**default_param_values, **kwargs, **param_group}
			# check that "lr" is defined (it should be either in kwargs, or in all of the param groups)
			assert "lr" in param_group, "Error: one of the parameter groups didn't specify a value for required parameter `lr`."
			# set the "params" and "gs" in param groups (note that we're storing 'gs' within each param group, rather than as self.gs)
			param_group["params"] = list(param_group["params"])
			param_group["gs"] = [t.zeros_like(p) for p in param_group["params"]]
			self.param_groups.append(param_group)
			# check that no params have been double counted
			for param in param_group["params"]:
				assert param not in params_to_check_for_duplicates, "Error: some parameters appear in more than one parameter group"
				params_to_check_for_duplicates.add(param)

		self.t = 1

	def zero_grad(self) -> None:
		for param_group in self.param_groups:
			for p in param_group["params"]:
				p.grad = None

	@t.inference_mode()
	def step(self) -> None:
		# loop through each param group
		for i, param_group in enumerate(self.param_groups):
			# get the parameters from the param_group
			lmda = param_group["weight_decay"]
			mu = param_group["momentum"]
			gamma = param_group["lr"]
			# loop through each parameter within each group
			for j, (p, g) in enumerate(zip(param_group["params"], param_group["gs"])):
				# Implement the algorithm in the pseudocode to get new values of params and g
				new_g = p.grad
				if lmda != 0:
					new_g = new_g + (lmda * p)
				if mu > 0 and self.t > 1:
					new_g = (mu * g) + new_g
				# Update params (remember, this must be inplace)
				param_group["params"][j] -= gamma * new_g
				# Update g
				self.param_groups[i]["gs"][j] = new_g
		self.t += 1



if MAIN:
	tests.test_sgd_param_groups(SGD)

# %% 2️⃣ WEIGHTS AND BIASES

def get_cifar(subset: int = 1):
	cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=IMAGENET_TRANSFORM)
	cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=IMAGENET_TRANSFORM)
	if subset > 1:
		cifar_trainset = Subset(cifar_trainset, indices=range(0, len(cifar_trainset), subset))
		cifar_testset = Subset(cifar_testset, indices=range(0, len(cifar_testset), subset))
	return cifar_trainset, cifar_testset


if MAIN:
	cifar_trainset, cifar_testset = get_cifar()
	
	imshow(
		cifar_trainset.data[:15],
		facet_col=0,
		facet_col_wrap=5,
		facet_labels=[cifar_trainset.classes[i] for i in cifar_trainset.targets[:15]],
		title="CIFAR-10 images",
		height=600
	)

# %%

@dataclass
class ResNetTrainingArgs():
	batch_size: int = 64
	epochs: int = 3
	optimizer: Type[t.optim.Optimizer] = t.optim.Adam
	learning_rate: float = 1e-3
	n_classes: int = 10
	subset: int = 10

# %%

class ResNetTrainer:
	def __init__(self, args: ResNetTrainingArgs):
		self.args = args
		self.model = get_resnet_for_feature_extraction(args.n_classes).to(device)
		self.optimizer = args.optimizer(self.model.out_layers[-1].parameters(), lr=args.learning_rate)
		self.trainset, self.testset = get_cifar(subset=args.subset)
		self.logged_variables = {"loss": [], "accuracy": []}

	def _shared_train_val_step(self, imgs: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
		imgs = imgs.to(device)
		labels = labels.to(device)
		logits = self.model(imgs)
		return logits, labels

	def training_step(self, imgs: Tensor, labels: Tensor) -> t.Tensor:
		logits, labels = self._shared_train_val_step(imgs, labels)
		loss = F.cross_entropy(logits, labels)
		loss.backward()
		self.optimizer.step()
		self.optimizer.zero_grad()
		return loss

	@t.inference_mode()
	def validation_step(self, imgs: Tensor, labels: Tensor) -> t.Tensor:
		logits, labels = self._shared_train_val_step(imgs, labels)
		classifications = logits.argmax(dim=1)
		n_correct = t.sum(classifications == labels)
		return n_correct

	def train_dataloader(self):
		self.model.train()
		return DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True)
	
	def val_dataloader(self):
		self.model.eval()
		return DataLoader(self.testset, batch_size=self.args.batch_size, shuffle=True)

	def train(self):
		progress_bar = tqdm(total=self.args.epochs * len(self.trainset) // self.args.batch_size)
		accuracy = t.nan

		for epoch in range(self.args.epochs):

			# Training loop (includes updating progress bar)
			for imgs, labels in self.train_dataloader():
				loss = self.training_step(imgs, labels)
				self.logged_variables["loss"].append(loss.item())
				desc = f"Epoch {epoch+1}/{self.args.epochs}, Loss = {loss:.2f}, Accuracy = {accuracy:.2f}"
				progress_bar.set_description(desc)
				progress_bar.update()

			# Compute accuracy by summing n_correct over all batches, and dividing by number of items
			accuracy = sum(self.validation_step(imgs, labels) for imgs, labels in self.val_dataloader()) / len(self.testset)

			self.logged_variables["accuracy"].append(accuracy.item())

# %%

if MAIN:
	args = ResNetTrainingArgs()
	trainer = ResNetTrainer(args)
	trainer.train()
	plot_train_loss_and_test_accuracy_from_trainer(trainer, title="Training ResNet on MNIST data")

# %%

def test_resnet_on_random_input(model: ResNet34, n_inputs: int = 3):
	indices = np.random.choice(len(cifar_trainset), n_inputs).tolist()
	classes = [cifar_trainset.classes[cifar_trainset.targets[i]] for i in indices]
	imgs = cifar_trainset.data[indices]
	device = next(model.parameters()).device
	with t.inference_mode():
		x = t.stack(list(map(IMAGENET_TRANSFORM, imgs)))
		logits: t.Tensor = model(x.to(device))
	probs = logits.softmax(-1)
	if probs.ndim == 1: probs = probs.unsqueeze(0)
	for img, label, prob in zip(imgs, classes, probs):
		display(HTML(f"<h2>Classification probabilities (true class = {label})</h2>"))
		imshow(
			img, 
			width=200, height=200, margin=0,
			xaxis_visible=False, yaxis_visible=False
		)
		bar(
			prob,
			x=cifar_trainset.classes,
			template="ggplot2", width=600, height=400,
			labels={"x": "Classification", "y": "Probability"}, 
			text_auto='.2f', showlegend=False,
		)


if MAIN:
	test_resnet_on_random_input(trainer.model)

# %%


import wandb

@dataclass
class ResNetTrainingArgsWandb(ResNetTrainingArgs):
	wandb_project: Optional[str] = 'day4-resnet'
	wandb_name: Optional[str] = None


class ResNetTrainerWandb:
	def __init__(self, args: ResNetTrainingArgsWandb):
		self.args = args
		self.model = get_resnet_for_feature_extraction(args.n_classes).to(device)
		self.optimizer = args.optimizer(self.model.out_layers[-1].parameters(), lr=args.learning_rate)
		self.trainset, self.testset = get_cifar(subset=args.subset)
		self.step = 0
		wandb.init(project=args.wandb_project, name=args.wandb_name, config=args)
		wandb.watch(self.model.out_layers[-1], log="all", log_freq=20)

	def _shared_train_val_step(self, imgs: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
		imgs = imgs.to(device)
		labels = labels.to(device)
		logits = self.model(imgs)
		return logits, labels

	def training_step(self, imgs: Tensor, labels: Tensor) -> t.Tensor:
		logits, labels = self._shared_train_val_step(imgs, labels)
		loss = F.cross_entropy(logits, labels)
		loss.backward()
		self.optimizer.step()
		self.optimizer.zero_grad()
		self.step += 1
		return loss

	@t.inference_mode()
	def validation_step(self, imgs: Tensor, labels: Tensor) -> t.Tensor:
		logits, labels = self._shared_train_val_step(imgs, labels)
		classifications = logits.argmax(dim=1)
		n_correct = t.sum(classifications == labels)
		return n_correct

	def train_dataloader(self):
		self.model.train()
		return DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True)
	
	def val_dataloader(self):
		self.model.eval()
		return DataLoader(self.testset, batch_size=self.args.batch_size, shuffle=True)

	def train(self):
		progress_bar = tqdm(total=self.args.epochs * len(self.trainset) // self.args.batch_size)
		accuracy = t.nan

		for epoch in range(self.args.epochs):

			# Training loop (includes updating progress bar)
			for imgs, labels in self.train_dataloader():
				loss = self.training_step(imgs, labels)
				wandb.log({"loss": loss.item()}, step=self.step)
				desc = f"Epoch {epoch+1}/{self.args.epochs}, Loss = {loss:.2f}, Accuracy = {accuracy:.2f}"
				progress_bar.set_description(desc)
				progress_bar.update()

			# Compute accuracy by summing n_correct over all batches, and dividing by number of items
			accuracy = sum(self.validation_step(imgs, labels) for imgs, labels in self.val_dataloader()) / len(self.testset)
			wandb.log({"accuracy": accuracy.item()}, step=self.step)

		wandb.finish()

# %%

if MAIN:
	args = ResNetTrainingArgsWandb()
	trainer = ResNetTrainerWandb(args)
	trainer.train()

# %%


if MAIN:
	sweep_config = dict()
	# FLAT SOLUTION
	# YOUR CODE HERE - fill `sweep_config`
	sweep_config = dict(
		method = 'random',
		metric = dict(name = 'accuracy', goal = 'maximize'),
		parameters = dict(
			batch_size = dict(values = [32, 64, 128, 256]),
			epochs = dict(min = 1, max = 4),
			learning_rate = dict(max = 0.1, min = 0.0001, distribution = 'log_uniform_values'),
		)
	)
	# FLAT SOLUTION END
	
	tests.test_sweep_config(sweep_config)

# %%

# (2) Define a training function which takes no args, and uses `wandb.config` to get hyperparams

class ResNetTrainerWandbSweeps(ResNetTrainerWandb):
	'''
	New training class made specifically for hyperparameter sweeps, which overrides the values in `args` with 
	those in `wandb.config` before defining model/optimizer/datasets.
	'''
	def __init__(self, args: ResNetTrainingArgsWandb):
		wandb.init(project=args.wandb_project, name=args.wandb_name)
		args.batch_size = wandb.config["batch_size"]
		args.epochs = wandb.config["epochs"]
		args.learning_rate = wandb.config["learning_rate"]
		self.args = args
		self.model = get_resnet_for_feature_extraction(args.n_classes).to(device)
		self.optimizer = args.optimizer(self.model.out_layers[-1].parameters(), lr=args.learning_rate)
		self.trainset, self.testset = get_cifar(subset=args.subset)
		self.step = 0
		wandb.watch(self.model.out_layers[-1], log="all", log_freq=20)


def train():
	args = ResNetTrainingArgsWandb()
	trainer = ResNetTrainerWandbSweeps(args)
	trainer.train()

# %%

if MAIN:
	sweep_id = wandb.sweep(sweep=sweep_config, project='part4-optimization-resnet-sweep')
	wandb.agent(sweep_id=sweep_id, function=train, count=3)
	wandb.finish()


# %%

