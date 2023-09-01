#%%
import os; os.environ['ACCELERATE_DISABLE_RICH'] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
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
from part1_transformer_from_scratch.answers_cleantransformer import Config, DemoTransformer

device = t.device("cuda:1" if t.cuda.is_available() else "cpu")
MAIN = __name__ == '__main__'

reference_gpt2 = HookedTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False,
                                                   device=device)
# %%
model_cfg = Config(
    debug=False, 
    d_model=768, 
    n_heads=12, 
    d_head=64, 
    d_mlp=3072, 
    n_layers=12, 
    n_ctx=1024, 
    d_vocab=reference_gpt2.cfg.d_vocab
)
model = DemoTransformer(model_cfg).to(device)
model.load_state_dict(reference_gpt2.state_dict(), strict=False)

tokenizer = reference_gpt2.tokenizer

class TransformerSampler:
    def __init__(self, model: DemoTransformer, tokenizer: GPT2TokenizerFast):
        self.model = model
        self.cfg = model.cfg
        self.tokenizer = tokenizer

    @t.inference_mode()
    def sample(self, prompt: str, max_tokens_generated=100, verbose=False, **kwargs) -> str:
        '''
            Returns a string of autoregressively generated text, starting from the prompt.
            Sampling terminates at max_tokens_generated, or when the model generates an
            end-of-sequence token.

            kwargs are passed to sample_next_token, to give detailed instructions on how
            new tokens are chosen    
        '''
        self.model.eval()
        tokens = self.tokenizer.encode(prompt, max_length=self.cfg.n_ctx)
        tokens = t.tensor([tokens], dtype=t.int, device=device) # (1, seq_len)
        #input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)[0] # (seq_len)

        for _ in range(max_tokens_generated):
            # Ensure this must be n_ctx
            tokens = tokens[:, -self.cfg.n_ctx:]
            logits = self.model(tokens) # shape (1, seq, d_vocab)
            # Get only the last logits
            logits = logits[0, -1]
            next_token: int = self.sample_next_token(tokens[0], logits, **kwargs)
            tokens = t.concat([tokens, t.tensor([[next_token]], device=device)], dim=-1) # (1, seq_len + 1)

            if next_token == self.tokenizer.eos_token_id:
                print(f"stop as {next_token} == {self.tokenizer.eos_token_id}")
                break
        
        output_str = self.tokenizer.decode(tokens[0])
        print(output_str)
        return output_str

    
    @t.inference_mode()
    def beam_search(
        self,
        prompt: str,
        num_return_sequences: int,
        num_beams:int ,
        max_new_tokens: int,
        no_repeat_ngram_size: int = 0,
        verbose=False
    ) -> List[Tuple[float, t.Tensor]]:
        '''
        Returns a string of autoregressively generated text, starting from the prompt.

        Sampling terminates at max_tokens_generated, or when the model generates an
        end-of-sequence token.

        kwargs are passed to sample_next_token, to give detailed instructions on how 
        new tokens are chosen.
        '''
        pass

    @staticmethod
    def sample_next_token(
        input_ids: Int[Tensor, "seq_len"],
        logits: Float[Tensor, "d_vocab"],
        temperature = 1.0,
        top_k=0,
        top_p = 0.0,
        frequency_penalty = 0.0,
        seed = None
    ):
        assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
        assert temperature >= 0, "Temperature should be non-negative"
        assert 0 <= top_p <= 1.0, "Top-p must be probability"
        assert 0 <= top_k, "Top-k must be non-negative"
        assert not (top_p != 0 and top_k != 0), "At most one of top-p and top-k supported"

        # set random seed for reproducibility
        if seed is not None:
            t.manual_seed(seed)
            np.random.seed(seed)

        # Apply all the specialized sampling methods
        if temperature == 0:
            return TransformerSampler.greedy_search(logits)
        elif temperature != 1.0:
            logits = TransformerSampler.apply_temperature(logits, temperature)

        if frequency_penalty != 0.0:
            logits = TransformerSampler.apply_frequency_penalty(input_ids, logits, frequency_penalty)
        if top_k > 0:
            return TransformerSampler.sample_top_k(logits, top_k)
        if top_p > 0.0:
            return TransformerSampler.sample_top_p(logits, top_p)

        return TransformerSampler.sample_basic(logits)
    
    @staticmethod
    def greedy_search(logits: Float[Tensor, "d_vocab"]) -> int:
        '''
            Return the most likely token (as an int)
        '''
        assert logits.ndim == 1
        out = logits.argmax().item()
        return out
    
    @staticmethod
    def apply_temperature(logits: Float[Tensor, "d_vocab"], temperature: float
            ) -> Float[Tensor, "d_vocab"]:
        '''
            Applies temperature scaling to the logits
        '''
        assert temperature > 0
        logits = logits / temperature
        return logits
    
    @staticmethod
    def apply_frequency_penalty(input_ids: Int[Tensor, "seq_len"], logits: Float[Tensor, "d_vocab"], freq_penalty: float) -> Float[Tensor, "d_vocab"]:
        '''
        Applies a frequency penalty to the logits.
        '''
        weight = t.ones_like(input_ids) * freq_penalty
        counts = t.bincount(input = input_ids, weights = weight, minlength=logits.size(0))
        outputs = logits - counts
        return outputs

    @staticmethod
    def sample_basic(logits: Float[Tensor, "d_vocab"]) -> int:
        '''
        Samples from the distribution defined by the logits.
        '''
        dist = t.distributions.categorical.Categorical(logits = logits)
        return dist.sample(t.Size([1])).item()

    @staticmethod
    def sample_top_k(logits: Float[Tensor, "d_vocab"], k: int) -> int:
        '''
        Samples from the top k most likely tokens.
        '''
        topk_word = t.topk(input = logits, k = k, dim=-1, largest=True)
        dist = t.distributions.categorical.Categorical(logits = topk_word.values)
        sample_indx = dist.sample().item()
        out = topk_word.indices[sample_indx].item()
        assert isinstance(out, int)
        return out

    @staticmethod
    def sample_top_p(logits: Float[Tensor, "d_vocab"], top_p: float, min_tokens_to_keep: int = 1) -> int:
        '''
        Samples from the most likely tokens which make up at least p cumulative probability.
        '''
        sorted_logits, ori_indices = t.sort(logits, dim=-1, descending=True, stable=True)
        prob = t.softmax(sorted_logits, dim=-1)
        probcum = t.cumsum(prob, dim=-1)
        # When we insert top_p before included_index, it is still sorted
        # ex: included_index = 5, then include [0,..,5] which is 6 elements
        included_index = t.searchsorted(probcum, top_p).item()
        # together with min_tokens
        final_inclusive_index = max(min_tokens_to_keep-1, included_index)
        # dist
        dist = t.distributions.categorical.Categorical(probs = prob[0:final_inclusive_index+1])
        # sample
        sample_indx = dist.sample().item()
        out = ori_indices[sample_indx].item()
        assert isinstance(out, int)

        return out
        


#%%
sampler = TransformerSampler(model, tokenizer)

prompt = "Jingle bells, jingle bells, jingle all the way"
print(f"Greedy decoding with prompt: {prompt!r}\n")

output = sampler.sample(prompt, max_tokens_generated=8, temperature=0.0)
print(f"Your model said: {output!r}\n")

expected = "Jingle bells, jingle bells, jingle all the way up to the top of the mountain."
assert output == expected

print("Tests passed!")
# %%
prompt = "John and Mary went to the"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
logits = model(input_ids)[0, -1]

expected_top_5 = {
    " church": 0.0648,
    " house": 0.0367,
    " temple": 0.0145,
    " same": 0.0104,
    " Church": 0.0097
}
frequency_of_top_5 = defaultdict(int)
N = 10_000

for _ in tqdm(range(N)):
    token = TransformerSampler.sample_next_token(input_ids.squeeze(), logits)
    frequency_of_top_5[tokenizer.decode(token)] += 1

for word in expected_top_5:
    expected_freq = expected_top_5[word]
    observed_freq = frequency_of_top_5[word] / N
    print(f"Word: {word!r:<9}. Expected freq {expected_freq:.4f}, observed freq {observed_freq:.4f}")
    assert abs(observed_freq - expected_freq) < 0.01, "Try increasing N if this fails by a small amount."

print("Tests passed!")

# %%
logits = t.tensor([1, 2]).log()

cold_logits = TransformerSampler.apply_temperature(logits, temperature=0.001)
print('A low temperature "sharpens" or "peaks" the distribution: ', cold_logits)
t.testing.assert_close(cold_logits, 1000.0 * logits)

hot_logits = TransformerSampler.apply_temperature(logits, temperature=1000.0)
print("A high temperature flattens the distribution: ", hot_logits)
t.testing.assert_close(hot_logits, 0.001 * logits)

print("Tests passed!")
# %%
bieber_prompt = "And I was like Baby, baby, baby, oh Like, Baby, baby, baby, no Like, Baby, baby, baby, oh I thought you'd always be mine, mine"
input_ids = tokenizer.encode(bieber_prompt, return_tensors="pt")
logits = t.ones(tokenizer.vocab_size)
penalized_logits = TransformerSampler.apply_frequency_penalty(input_ids.squeeze(), logits, 2.0)

assert penalized_logits[5156].item() == -11, "Expected 6 occurrences of ' baby' with leading space, 1-2*6=-11"
assert penalized_logits[14801].item() == -5, "Expected 3 occurrences of ' Baby' with leading space, 1-2*3=-5"

print("Tests passed!")
# %%
sampler = TransformerSampler(model, tokenizer)

N_RUNS = 1
your_prompt = "Jingle bells, jingle bells, jingle all the way"
cases = [
    ("High freq penalty", dict(frequency_penalty=100.0)),
    ("Negative freq penalty", dict(frequency_penalty=-3.0)),
    ("Too hot!", dict(temperature=2.0)),
    ("Pleasantly cool", dict(temperature=0.7)),
    ("Pleasantly warm", dict(temperature=0.9)),
    ("Too cold!", dict(temperature=0.01)),
]

table = Table("Name", "Kwargs", "Output", title="Sampling - Manual Testing")

for (name, kwargs) in cases:
    for i in range(N_RUNS):
        output = sampler.sample(your_prompt, max_tokens_generated=24, **kwargs)
        table.add_row(name, repr(kwargs), repr(output) + "\n")

rprint(table)
# %%
prompt = "John and Mary went to the"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
logits = model(input_ids)[0, -1]

expected_top_5 = {
    " church": 0.0648,
    " house": 0.0367,
    " temple": 0.0145,
    " same": 0.0104,
    " Church": 0.0097
}
topk_5_sum = sum(expected_top_5.values())

observed_freqs = defaultdict(int)

N = 10000
for _ in tqdm(range(N)):
    token = TransformerSampler.sample_next_token(input_ids.squeeze(), logits, top_k=5)
    observed_freqs[tokenizer.decode(token)] += 1

for word in expected_top_5:
    expected_freq = expected_top_5[word] / topk_5_sum
    observed_freq = observed_freqs[word] / N
    print(f"Word: {word!r:<9}. Expected freq = {expected_freq:.4f}, obsved freq = {observed_freq:.4f}")
    assert abs(observed_freq - expected_freq) < 0.015, "Try increasing N if this fails by a small amount"

# %%
sampler = TransformerSampler(model, tokenizer)
your_prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, "\
"previously unexplored valley, in the Andes Mountains. "\
"Even more surprising to the researchers was the fact that the unicorns spoke perfect English."

output = sampler.sample(your_prompt, temperature = 0.7, top_k = 40, max_tokens_generated = 64)
rprint(f"Your model said: \n\n[bold dark_orange]{output}")
# %%
prompt = "John and Mary went to the"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
logits = model(input_ids)[0, -1]

expected_top_10pct = {
    " church": 0.0648,
    " house": 0.0367, # These are the two most likely tokens, and add up to >10%
}
top_10pct_sum = sum(expected_top_10pct.values())

observed_freqs = defaultdict(int)

N = 10000
for _ in tqdm(range(N)):
    token = TransformerSampler.sample_next_token(input_ids.squeeze(), logits, top_p=0.1)
    observed_freqs[tokenizer.decode(token)] += 1

for word in expected_top_10pct:
    expected_freq = expected_top_10pct[word] / top_10pct_sum
    observed_freq = observed_freqs[word] / N
    print(f"Word: {word!r:<9}. Expected freq {expected_freq:.4f}, observed freq {observed_freq:.4f}")
    assert abs(observed_freq - expected_freq) < 0.01, "Try increasing N if this fails by a small amount."
# %%
a = t.tensor([1,2,5,-1,2], dtype=t.float)
b = t.softmax(a, dim=-1)
print(b)
x = t.searchsorted(b, 0.5, side='left')
print(x)
# %%
sampler = TransformerSampler(model, tokenizer)

your_prompt = "Eliezer Shlomo Yudkowsky (born September 11, 1979) is an American decision and artificial intelligence (AI) theorist and writer, best known for"
output = sampler.sample(your_prompt, temperature=0.7, top_p=0.95, max_tokens_generated=64)
rprint(f"Your model said:\n\n[bold dark_orange]{output}")
# %%
@dataclass
class Beams:
    '''
        Class to store beams during beam search
    '''
    model: DemoTransformer
    tokenizer: GPT2TokenizerFast
    logprob_sums: Float[Tensor, "batch"]
    tokens: Int[Tensor, "batch seq"]

    def new_beams(self, logprob_sums, tokens) -> "Beams":
        ''' Create a new Beams object with same model and tokenizer'''
        return Beams(self.model, self.tokenizer, logprob_sums, tokens)
    
    def __getitem__(self, idx) -> "Beams":
        '''Allows you to take a slice of the beams object along the batch dim'''
        return self.new_beams(self.logprob_sums[idx], self.tokens[idx])

    @property
    def logprobs_and_completions(self) -> List[Tuple[float, str]]:
        '''
            Returns self as a list of logprob sums and completions
            (useful for getting final output)
        '''
        return [
            (logprob_sum.item(), self.tokenizer.decode(tokens))
            for (logprob_sum, tokens) in zip(self.logprob_sums, self.tokens)
        ]

    def get_topk_non_repeating(
            self, logprobs: Float[Tensor, "batch d_vocab"],
            no_repeat_ngram_size: int, k: int) -> Tuple[Float[Tensor, "batch k"], Int[Tensor, "batch k"]]:
        '''
        logprobs: 
            tensor of the log-probs for the next token
        no_repeat_ngram_size:
            size of ngram to avoid repeating
        k:
            number of top logits to return, for each beam in our collection

        Returns:
            equivalent to the output of `logprobs.topk(dim=-1)`, but makes sure
            that no returned tokens would produce an ngram of size  `no_repeat_ngram_size`
            which has already appeared in `self.tokens`.
        '''

        # Step 1 is to generate ngrams for all beams
        ngrams = []
        beams, len = self.tokens.size()
        for i in range(len-no_repeat_ngram_size+1):
            start = i
            end = i+no_repeat_ngram_size
            ngrams.append(self.tokens[:, start:end])
        ngrams = t.stack(ngrams, dim=1) # (beams, len-ngrams, ngram)

        # Step 2 is get the last (beam, dvocab, ngrams) for the last ngram-1 and each d_vocab
        if no_repeat_ngram_size>1:
            repeated_tokens = einops.repeat(self.tokens[:, -no_repeat_ngram_size+1:], 
                                            'beams seq -> beams vocab seq', vocab = logprobs.size(-1))
            repeated_vocab = einops.repeat(t.arange(logprobs.size(-1), device=self.tokens.device), 
                        'vocab -> beam vocab 1', beam = beams)
            concat_tokens = t.concat([repeated_tokens, repeated_vocab], axis=-1) # (beams, vocab, ngrams)
        else:
            concat_tokens = einops.repeat(t.arange(logprobs.size(-1), device=self.tokens.device),
                        'vocab -> beam vocab 1', beam = beams)

        # Step 3. Important now is how to find which concat_tokens valid
        # 3.a Convert concat_tokens to (beams, vocab, 1, 2)
        concat_tokens = concat_tokens[:,:, None,:]
        # 3.b Convert ngrams to (beams, 1, len-grams, ngram)
        ngrams = ngrams[:, None, :,:]
        # What we do now is to find if each concat tokens is equal to **any** element in ngrams
        comparision = (concat_tokens == ngrams) # (beams, vocab, len-grams, ngram)
        comparision = t.all(comparision, axis=-1) # (beams, vocab, len-grams) # All because each token must be same
        mask = t.any(comparision, axis=-1) # (beams, vocab) # any because if vocab is equal to any len-grams, then it contains ngram
        # Replace logprobs with inf if mask == inf
        valid_logprobs = t.masked_fill(input = logprobs, mask = mask, value = -float("inf"))
        # topk with only valid numbers
        top_tokens, top_index = t.topk(input=valid_logprobs, k=k, dim=-1)

        return top_tokens, top_index

    def generate(self, toks_per_beam: int, 
                 no_repeat_ngrams_size: Optional[int] = None) -> "Beams":
        '''
        Starting from the current set of beams (which has length `num_beams`), returns a new
        set of `num_beams * toks_per_beam`, containing the best `toks_per_beam` continuations for each
        of the original beams.

        Optional argument `no_repeat_ngram_size` means your model won't generate any sequences with
        a repeating n-gram of this length.
        '''
        num_beams = self.tokens.size(0)

        logits = self.model(self.tokens) # (beams, seq, d_vocab)
        logits_logprob = t.log_softmax(input = logits[:, -1, :], dim=-1) # (beams, d_vocab)
        # Get only toks_per_beam
        if no_repeat_ngrams_size == None:
            chosen_logprobs, chosen_index = t.topk(logits_logprob, k=toks_per_beam) #(beams, toks_per_beam)
        else:
            chosen_logprobs, chosen_index = self.get_topk_non_repeating(logits_logprob,
                                        no_repeat_ngram_size=no_repeat_ngrams_size, k=toks_per_beam)

        chosen_logprobs_flatten = einops.rearrange(chosen_logprobs,
                                    "beam toks -> (beam toks)") #(beams * toks_per_beam)
        repeated_logprob_sum = einops.repeat(self.logprob_sums, 
                        'beam -> (beam x)', x = toks_per_beam) # (beams * toks_per_beam)
        added_logprob_sum = chosen_logprobs_flatten + repeated_logprob_sum

        # Sorted the add and get maximum
        new_tokens = t.concat([einops.repeat(self.tokens, "beam seq -> (beam k) seq", k = toks_per_beam), 
                               einops.rearrange(chosen_index, "beams toks -> (beams toks) 1")], dim=-1)

        return Beams(model, tokenizer, added_logprob_sum, new_tokens)


    def filter(self, num_beams: int) -> Tuple["Beams", "Beams"]:
        '''
        Returns:
            best_beams: Beams
                filtered version of self, containing all best `num_beams` which are also not terminated.

            early_terminations: Beams
                filtered version of self, containing all best `num_beams` which are also terminated.
                i.e. the sum of lengths of these two should equal `num_beams`.
        '''
        top_logprobs, top_index = t.topk(self.logprob_sums, k=num_beams, dim=-1)
        top_tokens = self.tokens[top_index]
        eos_token = tokenizer.eos_token_id

        early_terminated_mask = top_tokens[:, -1] == eos_token # (beams)
        
        early_termination_logprobs = top_logprobs[early_terminated_mask]
        early_termination_tokens = top_tokens[early_terminated_mask]

        best_beam_logprobs = top_logprobs[~early_terminated_mask]
        best_beam_tokens = top_tokens[~early_terminated_mask]

        best_beams = Beams(self.model, self.tokenizer, best_beam_logprobs, best_beam_tokens)
        early_terminations = Beams(self.model, self.tokenizer, early_termination_logprobs, 
                                  early_termination_tokens)
                
        return best_beams, early_terminations


    def print(self, title="Best completions", max_print_chars=80) -> None:
        '''
        Prints out a set of sequences with their corresponding logitsums.
        '''
        if len(self.tokens) == 0:
            return
        table = Table("logitsum", "completion", title=title)
        for logprob_sum, tokens in zip(self.logprob_sums, self.tokens):
            text = self.tokenizer.decode(tokens)
            if len(repr(text)) > max_print_chars:
                text = text[:int(0.3 * max_print_chars)] + " ... " + text[-int(0.7 * max_print_chars):]
            table.add_row(f"{logprob_sum:>8.3f}", repr(text))
        rprint(table)

@t.inference_mode()
def beam_search(
    self: TransformerSampler,
    prompt: str, 
    num_return_sequences: int, 
    num_beams: int, 
    max_new_tokens: int, 
    no_repeat_ngram_size: Optional[int] = None,
    verbose=False
) -> List[Tuple[float, Tensor]]:
    '''
    Implements a beam search, by repeatedly performing the `generate` and `filter` steps (starting
    from the initial prompt) until either of the two stopping criteria are met:

        (1) we've generated `max_new_tokens` tokens, or
        (2) we've generated `num_returns_sequences` terminating sequences.

    To modularize this function, most of the actual complexity is in the Beams class,
    in the `generate` and `filter` methods.
    '''

    assert num_return_sequences <= num_beams
    self.model.eval()

    tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(device) # (1, seq)
    logprob_sums = t.tensor([0.0], dtype=t.float).to(device)
    beams = Beams(self.model, self.tokenizer, logprob_sums, tokens)
    return_sequences = []

    for _ in range(max_new_tokens):
        beams = beams.generate(toks_per_beam=num_beams, no_repeat_ngrams_size=no_repeat_ngram_size)
        best_beams, terminated_beams = beams.filter(num_beams = num_beams)
        return_sequences.extend(terminated_beams.logprobs_and_completions)

        if len(return_sequences) >= num_return_sequences:
            return_sequences = return_sequences[:num_return_sequences]
            break

        beams = best_beams

    # If not enough return_sequences, add best_beams

    return_sequences.extend(best_beams.logprobs_and_completions)
    return_sequences = return_sequences[:num_return_sequences]

    return return_sequences



# %%
beams = Beams(
    model,
    tokenizer,
    logprob_sums=t.tensor([-10.0, -15.0, -20.0]).to(device),
    tokens = t.tensor([
        [5661, 318, 262, 2368],
        [5661, 318, 262, 1218],
        [5661, 318, 262, 717],
    ]).to(device)
)

beams.print()

print("Testing generate, without no_repeat_ngram_size argument:")
new_beams = beams.generate(toks_per_beam=2)
new_beams.print()
assert new_beams.logprobs_and_completions[0][1] == "this is the third time"
# %%
print("Testing generate, with no_repeat_ngram_size argument:")
bigram_beams = Beams(
    model, 
    tokenizer,
    logprob_sums = t.tensor([-0.0]).to(device),
    tokens = t.tensor([[530, 734, 530, 734]]).to(device)
    # tokens are " one two one two"
)

# with ngrams = 1, should not generat token " one" or " two"
new_bigram_beams = bigram_beams.generate(toks_per_beam=3, no_repeat_ngrams_size=1)
new_bigram_beams.print()
assert all([not (completion[1].endswith(" one") or completion[1].endswith(" two")) for 
            completion in new_bigram_beams.logprobs_and_completions])

new_bigram_beams = bigram_beams.generate(toks_per_beam=3, no_repeat_ngrams_size=2)
new_bigram_beams.print()
assert all([not completion[1].endswith(" one") for completion in new_bigram_beams.logprobs_and_completions])
assert any([not completion[1].endswith(" two") for completion in new_bigram_beams.logprobs_and_completions])

#%%
## Testing filters
logprob_sums = t.tensor([-1.0, -2.0]).to(device)
tokens = t.tensor([
    [19485, 13],
    [19485, tokenizer.eos_token_id]
]).to(device)

beams_with_eos = Beams(model, tokenizer, logprob_sums, tokens)
best_beams, early_terminations = beams_with_eos.filter(2)

t.testing.assert_close(best_beams.logprob_sums, logprob_sums[[0]])
t.testing.assert_close(best_beams.tokens, tokens[[0]])

assert early_terminations.logprobs_and_completions == [(-2.0, "Stop" + tokenizer.eos_token)]

print("All tests for `filter` passed!")

#%%
TransformerSampler.beam_search = beam_search
TransformerSampler.beam_search = beam_search

sampler = TransformerSampler(model, tokenizer)

prompt = "The ships hung in the sky in much the same way that"
orig_len = len(tokenizer.encode(prompt))

final_logitsums_and_completions = sampler.beam_search(
    prompt=prompt, 
    num_return_sequences=3,
    num_beams=40,
    max_new_tokens=60, 
    no_repeat_ngram_size=2,
    verbose=False
)

# Print all the best output
for logprob_sum, text in final_logitsums_and_completions:
    avg_logprob_as_prob = t.tensor(logprob_sum / (len(tokenizer.encode(text)) - orig_len)).exp().item()
    print("=" * 25 + f" Avg logprob (as probability) = {avg_logprob_as_prob:.3f} " + "=" * 25)
    rprint("Best output:\n\n[bold dark_orange]" + text)
# %%
array1 = t.tensor([[1,2],[3,4],[5,6],[1,3],[2,9]])
array2 = t.tensor([[1,3], [3,4]])
print(array1.size())
print(array2.size())
print(array1[:, None, :].size())
print(array2[None, :, :].size())
e = t.all(array1[:, None, :] == array2[None, :, :], axis=2)
print(e)
print(t.any(e, axis=1))
# %%
x = t.tensor([[1,2,3],[4,5,4],[3,44,3]])
t.topk(x, 100, dim=-1)
# %%
