#%%
import os; os.environ['ACCELERATE_DISABLE_RICH'] = "1"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import sys
import einops
from dataclasses import dataclass
from transformer_lens import HookedTransformer
from transformer_lens.utils import gelu_new, tokenize_and_concatenate
import torch as t
from torch import Tensor
import torch.nn as nn
import numpy as np
import math
from tqdm.notebook import tqdm
from typing import Tuple, List, Optional, Dict
from jaxtyping import Float, Int
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from collections import defaultdict
from rich.table import Table
from rich import print as rprint
import datasets
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from pathlib import Path
import webbrowser
import time

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part1_transformer_from_scratch").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow
# import part1_transformer_from_scratch.solutions as solutions

device = t.device("cuda:1" if t.cuda.is_available() else "cpu")
MAIN = __name__ == '__main__'

reference_gpt2 = HookedTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False,
                                                   device=device)
# %%
# Step 1. Convert text to tokens
reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer." \
    "One day I will exceed human level intelligence and take over the world!"
tokens = reference_gpt2.to_tokens(reference_text, prepend_bos=False).to(device)
print(tokens)
print(tokens.shape)
# Step 2. Map tokens to logits by running inference
logits, cache = reference_gpt2.run_with_cache(tokens)

# %%
# Printing activation
for activation_name, activation in cache.items():
    if ".0." in activation_name or "blocks" not in activation_name:
        print(f"{activation_name:30} {tuple(activation.shape)}")
# %%
# Printing parameters
for name, param in reference_gpt2.named_parameters():
    if ".0." in name or "blocks" not in name:
        print(f"{name:18} {tuple(param.shape)}")

# %%
@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12


cfg = Config()
print(cfg)
# %%
def rand_float_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    random_input = t.randn(shape).to(device)
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    if isinstance(output, tuple): output = output[0]
    print("Output shape:", output.shape, "\n")

def rand_int_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    random_input = t.randint(100, 1000, shape).to(device)
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    if isinstance(output, tuple): output = output[0]
    print("Output shape:", output.shape, "\n")

def load_gpt2_test(cls, gpt2_layer, input):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    layer.load_state_dict(gpt2_layer.state_dict(), strict=False)
    print("Input shape:", input.shape)
    output = layer(input)
    if isinstance(output, tuple): output = output[0]
    print("Output shape:", output.shape)
    try: reference_output = gpt2_layer(input)
    except: reference_output = gpt2_layer(input, input, input)
    print("Reference output shape:", reference_output.shape, "\n")
    comparison = t.isclose(output, reference_output, atol=1e-4, rtol=1e-3)
    #assert t.allclose(output, reference_output)
    print(f"{comparison.sum()/comparison.numel():.2%} of the values are correct\n")
# %%
class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(self, x: Float[Tensor, "batch seq dmodel"]):
        mean = t.mean(x, dim=-1, keepdim = True) # batch, seq, 1
        var = t.var(x, dim=-1, keepdim = True, unbiased=False) # batch, seq

        normalized = (x - mean)/ t.sqrt(var + self.cfg.layer_norm_eps) # batch, seq, dmodel
        output = einops.einsum(normalized, self.w, 'b s d, d -> b s d') + self.b

        return output

if MAIN:
    rand_float_test(LayerNorm, [2, 4, 768])
    load_gpt2_test(LayerNorm, reference_gpt2.ln_final, cache["resid_post", 11])
# %%
class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, mean=0, std=cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position dmodel"]:
        # Option 1: using advanced index
        output = self.W_E[tokens]

        return output

if MAIN: 
    rand_int_test(Embed, [2, 4])
    load_gpt2_test(Embed, reference_gpt2.embed, tokens)    
# %%
class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, mean=0, std=cfg.init_range)

    def forward(self, tokens: [Int, Tensor, "batch position"]) -> Float[Tensor, "batch position dmodel"]:
        tokens_index = t.arange(tokens.size(1), device=tokens.device)
        tokens_index = einops.repeat(tokens_index, 'c -> b c', b = tokens.size(0))
        # Option 1: Using advanced indexing
        output = self.W_pos[tokens_index]

        # option 2: matrix multiplication
        eye_matrix = t.eye(n = self.cfg.n_ctx, device=tokens.device, dtype=t.int64)
        onehot = eye_matrix[tokens_index] # batch, pos, n_ctx
        output2 = einops.einsum(onehot.float(), self.W_pos, 'b p c, c d-> b p d')

        # Option 3: basic indexing
        output3 = einops.repeat(self.W_pos[:tokens.size(1)], 'pos d -> b pos d', b = tokens.size(0))

        assert t.allclose(output, output2)
        assert t.allclose(output, output3)
        return output2
if MAIN:
    rand_int_test(PosEmbed, [2, 4])
    load_gpt2_test(PosEmbed, reference_gpt2.pos_embed, tokens)
# %%
class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(t.empty(size=(cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty(size=(cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty(size=(cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.b_Q = nn.Parameter(t.zeros(size=(cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros(size=(cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros(size=(cfg.n_heads, cfg.d_head)))

        self.W_O = nn.Parameter(t.empty(size=(cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.b_O = nn.Parameter(t.zeros(size=(cfg.d_model,)))

        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        #nn.init.normal_(self.b_Q, std=self.cfg.init_range)
        #nn.init.normal_(self.b_K, std=self.cfg.init_range)
        #nn.init.normal_(self.b_V, std=self.cfg.init_range)

        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        #nn.init.normal_(self.b_O, std=self.cfg.init_range)
        
        self.register_buffer("IGNORE", t.tensor(-1e5, dtype=t.float32, device=device))

    def forward(self, normalized_resid_pre: Float[Tensor, "batch pos d_model"]
                ) -> Float[Tensor, "batch pos d_model"]:
        query = einops.einsum(normalized_resid_pre, self.W_Q, 
                              "batch pos d_model, n_heads d_model d_head -> batch pos n_heads d_head") + self.b_Q
        key = einops.einsum(normalized_resid_pre, self.W_K, 
                            "batch pos d_model, n_heads d_model d_head -> batch pos n_heads d_head") + self.b_K
        weight = einops.einsum(query, key, 
                               "batch query n_heads d_head, batch key n_heads d_head -> batch n_heads query key") 
        weight = weight / math.sqrt(self.cfg.d_head)

        attention = self.apply_casual_mask(weight)

        value = einops.einsum(normalized_resid_pre, self.W_V, 
                              "batch pos d_model, n_heads d_model d_head -> batch pos n_heads d_head") + self.b_V
        z = einops.einsum(attention, value, "batch n_heads pos_Q pos_K, batch pos_K n_heads d_head -> batch pos_Q n_heads d_head")

        output = einops.einsum(z, self.W_O,
                               "batch pos n_heads d_head, n_heads d_head d_model -> batch pos d_model") + self.b_O
        
        return output

    def apply_casual_mask(self, att_scores: Float[Tensor, "batch n_heads query_pos key_pos"]
                          ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        """
            Applies a casual mask to attention scores, and return masked scores
        """
        ### Given inputs [[1,2,3], [4,5,6], [7,8,9]]
        ### You want [[1, 0, 0], [0.5, 0.5, 0], [0.3, 0.3, 0.3]] # some thing like that
        ### Option 1, using masked_fill_
        ## Because masked_fill_ fills 1's element, thus we use torch.triu to have upper triangular
        ones = t.ones_like(att_scores, device=att_scores.device) #[[1,1,1],[1,1,1],[1,1,1]]
        mask = t.triu(ones, diagonal=1) #[[0,1,1],[0,0,1],[0,0,0]]
        # [[val, -inf, -inf], [val, val, -inf], [val, val, val]], val depends on att_Scores
        attention_scores = t.masked_fill(input = att_scores, mask = mask.bool(), value = self.IGNORE) 
        attention_scores = t.softmax(attention_scores, dim=-1)  # normalize
        #print('hihihi')
        #print(attention_scores)
        ## Or you can use t.where
        another_scores = t.where(mask.bool(), self.IGNORE, att_scores)
        another_scores = t.softmax(another_scores, dim=-1)
        #print('hahah')
        #print(another_scores)
        #print('---')

        assert t.allclose(attention_scores, another_scores)
        # Option 2 using lower triangular.
        ## However with lower triangular, we cannot use masked_fill because it requires a 
        # 'true' mask to be filled witl IGNORE
        mask = t.tril(t.ones_like(att_scores), diagonal=0) # use 0 as we keep diagonal
        scores = t.where(mask.bool(), att_scores, self.IGNORE)
        scores = t.softmax(scores, dim=-1)

        assert t.allclose(scores, attention_scores)

        return attention_scores
if MAIN:
    rand_float_test(Attention, [2, 4, 768])
    load_gpt2_test(Attention, reference_gpt2.blocks[0].attn, cache["normalized", 0, "ln1"])

# %%
class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
        self.W_out = nn.Parameter(t.empty(cfg.d_mlp, cfg.d_model))
        self.b_in = nn.Parameter(t.zeros((cfg.d_mlp)))
        self.b_out = nn.Parameter(t.zeros((cfg.d_model)))

        nn.init.normal_(self.W_in, std=cfg.init_range)
        nn.init.normal_(self.W_out, std=cfg.init_range)

    def forward(self, normalized_resid_mid: Float[Tensor, "batch pos d_model"]
                ) -> Float[Tensor, "batch pos d_model"]:
        pre = einops.einsum(normalized_resid_mid, self.W_in, 
                            "batch pos d_model, d_model d_mlp -> batch pos d_mlp") + self.b_in
        post = t.nn.functional.gelu(pre, approximate='tanh')

        out = einops.einsum(post, self.W_out,
                            "batch pos d_mlp, d_mlp d_model -> batch pos d_model") + self.b_out
        return out

if MAIN:
    rand_float_test(MLP, [2, 4, 768])
    load_gpt2_test(MLP, reference_gpt2.blocks[0].mlp, cache["normalized", 0, "ln2"])
# %%
class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(
        self, resid_pre: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_model"]:
        
        ln1_output = self.ln1(resid_pre)
        attn_out = self.attn(ln1_output)

        resid_mid = attn_out + resid_pre

        assert ~t.isnan(resid_mid).any(), "nan at resid mid"

        ln2_output = self.ln2(resid_mid)
        mlp_output = self.mlp(ln2_output)

        resid_post = mlp_output + resid_mid

        assert ~t.isnan(resid_post).any(), "nan at resid post"

        return resid_post

if MAIN:
    rand_float_test(TransformerBlock, [2, 4, 768])
    load_gpt2_test(TransformerBlock, reference_gpt2.blocks[0], cache["resid_pre", 0])
# %%
class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty(size=(cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std = self.cfg.init_range)
        self.b_U = nn.Parameter(t.zeros((cfg.d_vocab)), requires_grad=False)

    def forward(self, normalized_resid_final: Float[Tensor, "batch pos d_model"]
                    ) -> Float[Tensor, "batch pos d_vocab"]:
        return einops.einsum(normalized_resid_final, self.W_U, 
                             "batch pos d_model, d_model d_vocab -> batch pos d_vocab") + self.b_U

if MAIN: 
    rand_float_test(Unembed, [2, 4, 768])
    load_gpt2_test(Unembed, reference_gpt2.unembed, cache["ln_final.hook_normalized"])
# %%
class DemoTransformer(nn.Module):
    def __init__(self, cfg:Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(self, tokens: Int[Tensor, "batch pos"]) -> Float[Tensor, "batch pos d_vocab"]:
        embed = self.embed(tokens)
        pos_embed = self.pos_embed(tokens)
        residual = embed + pos_embed

        for block in self.blocks:
            output = block(residual)
            residual = output
        output = self.ln_final(output)
        logits = self.unembed(output)

        return logits

if MAIN:
    rand_int_test(DemoTransformer, [2, 4])
    load_gpt2_test(DemoTransformer, reference_gpt2, tokens)
# %%
demo_gpt2 = DemoTransformer(Config(debug=False)).to(device)
demo_gpt2.load_state_dict(reference_gpt2.state_dict(), strict=False)

demo_logits = demo_gpt2(tokens)
print(demo_logits)
# %%
def get_log_probs(logits: Float[Tensor, "batch pos d_vocab"], labels: Int[Tensor, "batch pos"]):
    # labels: A B C D -> predict logits B C D E. 
    # Thus we trim first pos of labels and last pos of logits
    log_probs = logits.log_softmax(dim=-1)
    return t.gather(log_probs[:, :-1, :], dim=2, index=labels[:, 1:].unsqueeze(-1)).squeeze(-1)
     
pred_log_probs = get_log_probs(demo_logits, tokens)
print(f"Avg cross entropy loss: {-pred_log_probs.mean():.4f}")
print(f"Avg cross entropy loss for uniform distribution: {math.log(demo_gpt2.cfg.d_vocab):4f}")
print(f"Avg probability assigned to correct token: {pred_log_probs.exp().mean():4f}")
# %%

######## Training Transformer #############
model_cfg = Config(
    debug=False, 
    d_model=256, 
    n_heads=4, 
    d_head=64, 
    d_mlp=1024, 
    n_layers=2, 
    n_ctx=256, 
    d_vocab=reference_gpt2.cfg.d_vocab
)
model = DemoTransformer(model_cfg)

@dataclass
class TransformerTrainingArgs():
    batch_size = 16
    epochs = 600
    max_steps_per_epoch = 200
    lr = 1e-3
    weight_decay = 1e-2
    wandb_project: Optional[str] = "day1-demotransformer"
    wandb_name: Optional[str] = None
    use_wandb = True

args = TransformerTrainingArgs()

## Creating data
if MAIN:
    dataset = datasets.load_dataset("NeelNanda/pile-10k", split="train").remove_columns("meta")
    print(dataset)
    print(dataset[0]['text'][:100])
    ### Tokenized
    tokenized_dataset = tokenize_and_concatenate(dataset, reference_gpt2.tokenizer, streaming=False,
                                                max_length = model.cfg.n_ctx, column_name="text",
                                                add_bos_token=True, num_proc=4)
    dataset_dict = tokenized_dataset.train_test_split(test_size=1000)
    train_loader = DataLoader(dataset_dict["train"],
                            batch_size=args.batch_size, shuffle=True, num_workers=4,
                            pin_memory=True)
    test_loader = DataLoader(dataset_dict["test"],
                            batch_size=args.batch_size, shuffle=False, num_workers=4,
                            pin_memory=True)

# %%
class TransformerTrainer:
    def __init__(self, args: TransformerTrainingArgs, model: DemoTransformer):
        self.args = args
        self.model = model
        self.optimizer = t.optim.AdamW(params = self.model.parameters(),
                                       lr = args.lr,
                                       weight_decay=args.weight_decay)
        self.step = 0

        if self.args.use_wandb:
            wandb.init(config = self.args, project = self.args.wandb_project, name=self.args.wandb_name)

    def training_step(self, batch: Dict[str, Int[Tensor, "batch seq"]]) -> Tuple[Float[Tensor, ""]]:
        '''
		Calculates the loss on the tokens in the batch, performs a gradient update step, and logs the loss.

		Remember that `batch` is a dictionary with the single key 'tokens'.
		'''
        # YOUR CODE HERE
        self.optimizer.zero_grad()

        # (batch, seq)  
        tokens = batch["tokens"].to(device)
        # (batch, seq, d_vocab)
        logits = self.model(tokens)

        # (batch, seq, d_vocab)
        log_probs = logits.log_softmax(dim=-1)
        # (batch, seq)
        log_probs_for_tokens = t.gather(input = log_probs[:, :-1], dim=2, 
                                        index = tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
        loss = -log_probs_for_tokens.mean()
        loss.backward()

        self.optimizer.step()

        # Recording
        # mean over the sequence then mean over batch, it is actually same as .mean()
        running_loss = -log_probs_for_tokens.mean(dim=-1).mean().item()

        predicted_positions = log_probs.argmax(dim=-1) # (batch, seq)
        running_acc = (predicted_positions[:, :-1] == tokens[:, 1:]).float().mean().item()

        self.step+=1

        return running_loss, running_acc


    def validation_step(self, batch: Dict[str, Int[Tensor, "batch seq"]]) -> Tuple[Float[Tensor, ""]]:
        '''
        Calculates & returns the accuracy on the tokens in the batch (i.e. how often the model's prediction
		is correct). Logging should happen in the `train` function (after we've computed the accuracy for 
		the whole validation set).
        '''
        # (batch, seq)  
        tokens = batch["tokens"].to(device)
        # (batch, seq, d_vocab)
        logits = self.model(tokens)

        # (batch, seq, d_vocab)
        log_probs = logits.log_softmax(dim=-1)
        # (batch, seq)
        log_probs_for_tokens = t.gather(input = log_probs[:, :-1], dim=2, 
                                        index = tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
        # Recording
        # sum over the sequence then mean over batch
        val_loss = -log_probs_for_tokens.mean(dim=-1).mean().item()

        predicted_positions = log_probs.argmax(dim=-1) # (batch, seq)
        val_acc = (predicted_positions[:, :-1] == tokens[:, 1:]).float().mean().item()

        return val_loss, val_acc

    def train(self):
        '''
		Trains the model, for `self.args.epochs` epochs. Also handles wandb initialisation, and early stopping
		for each epoch at `self.args.max_steps_per_epoch` steps.
		'''
        train_loader = self.train_loader()
        test_loader = self.test_loader()

        for epoch in range(self.args.epochs):
            train_loss, train_acc = 0, 0
            val_loss, val_acc = 0, 0
            running_loss, running_acc = 0, 0
            # put to train mode
            start = time.time()
            self.model.train()
            for index, train_batch in enumerate(train_loader):
                per_loss, per_acc = self.training_step(train_batch)

                train_loss += per_loss * train_batch["tokens"].size(0)
                train_acc += per_acc * train_batch["tokens"].size(0)
                running_loss += per_loss
                running_acc += per_acc

                #if index > self.args.max_steps_per_epoch:
                    #break

                if (index + 1) % 5:
                    print(f"Train: Epoch {epoch}|{self.args.epochs}." \
                        f" Done {100*(index+1)/len(train_loader):.2f}%. " \
                        f" Time {(time.time() - start):.2f} elapsed"\
                        f" Running Loss: {running_loss/(index+1):.2f} Running Acc {running_acc/(index+1):.2f} ")

                if self.args.use_wandb:
                    wandb.log({"running_acc": running_acc /(index+1),
                            "running_loss":running_loss / (index+1)}, step=self.step)

            # put to eval mode
            self.model.eval()
            running_loss, running_acc = 0, 0
            for index, test_batch in enumerate(test_loader):
                per_loss, per_acc = self.validation_step(test_batch)
                val_loss += per_loss * test_batch["tokens"].size(0)
                val_acc += per_acc * test_batch["tokens"].size(0)

                running_loss += per_loss
                running_acc += per_acc


                if (index + 1) % 5:
                    print(f"Val: Epoch {epoch}|{self.args.epochs}." \
                        f" Done {100*(index+1)/len(test_loader):.2f}%. " \
                        f" Time {(time.time() - start):.2f} elapsed"\
                        f" Running Loss: {running_loss/(index+1):.2f} Running Acc {running_acc/(index+1):.2f} ")

            train_loss = train_loss / len(train_loader.dataset)
            train_acc = train_acc / len(train_loader.dataset)
            val_loss = val_loss / len(test_loader.dataset)
            val_acc = val_acc / len(test_loader.dataset)

            if self.args.use_wandb:
                wandb.log({"train_loss" : train_loss, "train_acc":train_acc,
                       "val_loss": val_loss, "val_acc": val_acc}, step = epoch)
                
            print(f"Epoch {epoch}/{self.args.epochs}."\
                f"Train loss: {train_loss} train acc: {train_acc} " \
                f"Val loss: {val_loss} val acc: {val_acc}")

    def train_loader(self) -> DataLoader:
        '''
            Returns train loader (as in code above)
        '''
        return DataLoader(dataset_dict["train"], batch_size = self.args.batch_size,
                          shuffle = True, num_workers=4, pin_memory=True)
    
    def test_loader(self) -> DataLoader:
        '''
            Returns test loader
        '''
        return DataLoader(dataset_dict["test"], batch_size=self.args.batch_size,
                          shuffle = False, num_workers=4, pin_memory=True)

# %%
if MAIN:
    args = TransformerTrainingArgs()
    model = DemoTransformer(cfg=model_cfg).to(device)
    trainer = TransformerTrainer(args, model)
    trainer.train()

# %%
