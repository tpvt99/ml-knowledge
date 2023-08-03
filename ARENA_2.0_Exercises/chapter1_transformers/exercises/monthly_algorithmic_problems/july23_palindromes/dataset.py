import torch as t
from torch.utils.data import Dataset
from rich import print as rprint
from jaxtyping import Int, Float
from typing import Optional
from torch import Tensor

class PalindromeDataset(Dataset):

    def __init__(self, size, max_value, half_length, seed=42):
        '''
        We create our non-palindromic examples via the following process (for each sequence):

            1. Generate a random seq of length N/2
            2. Generate another random seq of length N/2, by randomly changing X values of the previous
               seq, where X is some random integer between 0 and N/2 inclusive.
            3. Concatenate the two sequences (flipping the second one)
        
        This makes sure we have a good variety of palindromic numbers, including quite a lot with only
        one number flipped (otherwise it would be too easy for the model to distinguish).
        '''
        assert size % 2 == 0
        self.max_value = max_value
        self.size = size
        self.half_length = half_length
        self.length = 2 * half_length
        t.manual_seed(seed)  # for reproducible results

        self.START = max_value + 1
        self.END = max_value + 2

        half_sequences = t.randint(low=0, high=max_value+1, size=(size//2, half_length))
        half_sequences_random = t.randint(low=0, high=max_value+1, size=(size//2, half_length))
        positions_to_flip = t.randint(low=0, high=2, size=(size//2, half_length)).bool()
        non_palindromic_sequences = t.concat([
            half_sequences,
            t.where(positions_to_flip, half_sequences_random, half_sequences.flip(-1))
        ], dim=1)

        half_sequences = t.randint(low=0, high=max_value+1, size=(size//2, half_length))
        palindromic_sequences = t.concat([
            half_sequences,
            half_sequences.flip(-1)
        ], dim=1)

        all_sequences = t.concat([palindromic_sequences, non_palindromic_sequences], dim=0)
        all_sequences = all_sequences[t.randperm(size)]
        assert all_sequences.shape == (size, self.length)

        self.is_palindrome = (all_sequences[:, :half_length] == all_sequences[:, half_length:].flip(-1)).all(-1).long()

        self.toks = t.concat([
            t.ones((size, 1), dtype=t.long) * self.START,
            all_sequences,
            t.ones((size, 1), dtype=t.long) * self.END,
        ], dim=-1)

        self.str_toks = []
        for tok in self.toks:
            self.str_toks.append(["START"] + [f"{t:02}" for t in tok[1:-1]] + ["END"])


    def __getitem__(self, index):
        return self.toks[index], self.is_palindrome[index]

    def __len__(self):
        return self.size

    def to(self, device: str):
        self.toks = self.toks.to(device)
        self.is_palindrome = self.is_palindrome.to(device)
        return self


def display_seq(toks: Int[Tensor, "seq"], prob_palindrome: Optional[Float] = None, dark_mode: bool = True) -> None:
    '''
    Displays a sequence of tokens, highlighting non-palindromic tokens orange.

    Also optionally prints the probability of the sequence being a palindrome.

    This was created for dark mode; you can change the colors by replacing 'dark_orange' or 'white' below.
    '''
    is_palindromic = toks[1:-1] == toks[1:-1].flip(0)

    color, highlight_color = ("white", "dark_orange") if dark_mode else ("black", "red")

    s = ("START |" + "|".join(
        f"[{highlight_color} bold]{tok:02}[/]" if not(is_p) else f"[{color}]{tok:02}[/]"
        for tok, is_p in zip(toks[1:-1], is_palindromic)
    ) + "| END")

    if prob_palindrome is not None: s += f"  ->  {prob_palindrome:.3f}"

    rprint(s)
