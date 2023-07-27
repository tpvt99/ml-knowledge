#%%

import os
import sys
import numpy as np
import einops
from typing import Union, Optional, Tuple
import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
import functools
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm.notebook import tqdm

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part2_cnns"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, line, bar
from part2_cnns.utils import display_array_as_img, display_soln_array_as_img
import part2_cnns.tests as tests

MAIN = __name__ == "__main__"
# %%
arr = np.load(section_dir / "numbers.npy")
if MAIN:
    display_array_as_img(arr[0])
# %%

### Exercise 1
arr1 = einops.rearrange(arr, 'n c h w -> c h (n w)')
if MAIN:
    display_array_as_img(arr1)

# %%

### Exercise 2
arr2 = einops.repeat(arr[0], 'c h w -> 2 c h w')
arr2 = einops.rearrange(arr2, 'b c h w -> c (b h) w')
if MAIN:
    display_array_as_img(arr2)

# %%

### Exercise 3
arr3 = einops.repeat(arr[0:2], 'b c h w -> c (b h) (2 w)')
if MAIN:
    display_array_as_img(arr3)

# %%

arr4 = einops.repeat(arr[0], 'c h w -> c (h 2) w')
if MAIN:
    display_array_as_img(arr4)
# %%

arr5 = einops.repeat(arr[0], 'c h w -> (c 3) h w')
arr5 = einops.rearrange(arr5, '(k c) h w -> c h (k w)', k = 3)
if MAIN:
    display_array_as_img(arr5)
# %%

### Exercise 6

arr6 = einops.rearrange(arr, '(b2 b1) c h w -> c (b2 h) (b1 w)', b1 = 3, b2 = 2)
if MAIN:
    display_array_as_img(arr6)

# %%

### Exerise 7
arr7 = einops.rearrange(arr, 'b c h w -> h (b w)')
if MAIN:
    display_array_as_img(arr7)
# %%
