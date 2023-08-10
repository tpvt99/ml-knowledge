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

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part1_transformer_from_scratch").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow
# import part1_transformer_from_scratch.solutions as solutions

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == '__main__'

reference_gpt2 = HookedTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)
# %%
sorted(list(reference_gpt2.tokenizer.vocab.items()), key = lambda x: x[1])
# %%
# Step 1. Convert text to tokens
reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer." \
    "One day I will exceed human level intelligence and take over the world!"
tokens = reference_gpt2.to_tokens(reference_text, prepend_bos=False).to(device)
print(tokens)
print(tokens.shape)
# Step 2. Map tokens to logits by running inference
logits, cache = reference_gpt2.run_with_cache(tokens)
print(f"Logit shape: {logits.shape}")
# Step 3. COnvert logits to softmax
probs = logits.softmax(dim=-1)
print(f"Probs shape {probs.shape}")
most_likely_next_tokens = probs.argmax(dim=-1)[0]
most_likely_tokens_using_logits = logits.argmax(dim=-1)
print(most_likely_next_tokens) # same below
print(most_likely_tokens_using_logits) # same as above
print(list(zip(reference_gpt2.to_str_tokens(tokens), 
               reference_gpt2.tokenizer.batch_decode(most_likely_next_tokens))))
# Step 4. Map distribution to a token
next_token = logits[0, -1].argmax(dim=-1)
next_char = reference_gpt2.to_string(next_token)
print(f"next token {next_token} next char {next_char}")
# %%
# Fully step: Running to generate text
print(f"Sequence so far: {reference_gpt2.to_string(tokens)[0]!r}")
print(f"Tokens shape {tokens.shape}")
for i in range(10):
    # predicted:
    logits, cache = reference_gpt2.run_with_cache(tokens)
    # next token
    next_token = t.argmax(logits[[0], -1], dim=-1, keepdim=True)
    # Concat next_token to tokens
    updated_tokens = t.concat([tokens, next_token], dim=-1)
    print(f"Sequence so far: {reference_gpt2.to_string(updated_tokens)}")
    #print(f"Tokens shape {updated_tokens.shape}")
    tokens = updated_tokens
# %%
print(f"Sequence so far: {reference_gpt2.to_string(tokens)[0]!r}")

for i in range(10):
    print(f"{tokens.shape[-1]+1}th char = {next_char!r}")
    # Define new input sequence, by appending the previously generated token
    tokens = t.cat([tokens, next_token[None, None]], dim=-1)
    # Pass our new sequence through the model, to get new output
    logits = reference_gpt2(tokens)
    # Get the predicted token at the end of our sequence
    next_token = logits[0, -1].argmax(dim=-1)
    # Decode and print the result
    next_char = reference_gpt2.to_string(next_token)
# %%
