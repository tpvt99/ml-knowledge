#%%
import sys
from pathlib import Path

import einops
import numpy as np
import torch as t
from torch import Tensor

# Make sure exercises are in path
chapter = "chapter0_fundamentals"
section = "part0_prereqs"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))


import part0_prereqs.tests as tests
from part0_prereqs.utils import display_array_as_img, display_soln_array_as_img

MAIN = __name__ = "__main__"
# %%
arr = np.load(section_dir / "numbers.npy")
print(arr.shape) # there are 6 images, shape (6, 3, 150, 150)
print(arr[0].shape)
display_array_as_img(arr[0]) # plotting first image in the batch
print(arr[0, 0].shape)
display_array_as_img(arr[0, 0])
display_array_as_img(arr[0, 1])
display_array_as_img(arr[0, 2])

# Start using einops
arr_stacked = einops.rearrange(arr, "b c h w -> c h (b w)")
#arr_stacked = einops.rearrange(arr, "b c h w -> c w (b h)") # rotate -90
print(arr_stacked.shape)
display_array_as_img(arr_stacked)
# %% 
# Column Stacking
arr1 = einops.rearrange(arr, "b c h w -> c (b h) w")
print(arr1.shape)
display_array_as_img(arr1)

# %%
# Column-stacking and copying
arr2 = einops.repeat(arr[0], "c h w -> c (repeat h) w", repeat=2)
#arr2 = einops.repeat(arr[0], "c h w -> c (2 h) w") same
print(arr2.shape)
display_array_as_img(arr2)

# %%
# Row-stacking and double-copying
print(arr[0:2].shape)
arr3_temp = einops.rearrange(arr[0:2], "b c h w -> c (b h) w") # (h b) and (b h) different
display_array_as_img(arr3_temp)
arr3 = einops.repeat(arr3_temp, "c h w -> c h (2 w)") # same here (w 2) different (2 w)
display_array_as_img(arr3)

# Or in 1 operations
arr3 = einops.repeat(arr[0:2], "b c h w -> c (b h) (2 w)")
display_array_as_img(arr3)
# %%
# Stretching
arr4 = einops.repeat(arr[0], "c h w -> c (h 2) w") # stretching
#arr4 = einops.repeat(arr[0], "c h w -> c (2 h) w") # duplicate
display_array_as_img(arr4)
# %%
# Split channels
arr5 = einops.rearrange(arr[0], "c h w -> h (c w)")
display_array_as_img(arr5)
# %%
# Stack into rows and cols
arr6 = einops.rearrange(arr, "(b1 b2) c h w -> c (b1 h) (b2 w)", b1=2)
display_array_as_img(arr6)
# %%
# Tranpose
arr7 = einops.rearrange(arr[1], "c h w -> c w h")
display_array_as_img(arr7)
# %%
arr8 = einops.reduce(arr, "(b1 b2) c (h1 h2) (w1 w2) -> c (b1 h1) (b2 w1)", "max", h1=2, w1=2, b1=2)
display_array_as_img(arr8)
# %%
# Broadcasting
x = t.ones((4, 1))
y = t.ones((4))
print(y.shape)
z = x + y
t.arange(3, 9)
# %%
def rearrange_1() -> Tensor:
    """Return the following tensor using only t.arange and einops.rearrange:
    
   [[3, 4],
    [5, 6],
    [7, 8]] 
    """
    return einops.rearrange(t.arange(3, 9), "(w1 w2) -> w1 w2", w1 = 3)

tests.test_rearrange_1(rearrange_1)

def rearrange_2() -> Tensor:
    """Return the following tensor using only t.arange and einops.rearrange:
    
   [[1, 2, 3],
    [4, 5, 6]]
    """
    return einops.rearrange(t.arange(1, 7), "(h w) -> h w", h = 2, w = 3)

tests.test_rearrange_2(rearrange_2)

def temperatures_average(temps: Tensor) -> Tensor:
    """Return the average temperature for each week.

    temps: a 1D temperature containing temperatures for each day.
    Length will be a multiple of 7 and the first 7 days are for the first week, second 7 days for the second week, etc.

    You can do this with a single call to reduce.
    """
    assert len(temps) % 7 == 0
    return einops.reduce(temps, "(weeks days) -> weeks", "mean", days=7)

tests.test_temperatures_average(temperatures_average)

def temperatures_differences(temps: Tensor) -> Tensor:
    """For each day, subtract the average for the week the days belong to.
    
    temps: as above
    """
    temp_avg = temperatures_average(temps)
    temp_avg_repeat = einops.repeat(temp_avg, "a -> (a k)", k = 7)
    return temps - temp_avg_repeat

tests.test_temperatures_differences(temperatures_differences)

def temperatures_normalized(temps: Tensor) -> Tensor:
    """For each day, subtract the weekly average and divide by the weekly standard deviation.

    temps: as above

    Pass t.std to reduce.
    """
    temp_avg = einops.reduce(temps, "(a 7) -> a", "mean") # (2)
    temp_std = einops.reduce(temps, "(a 7) -> a", t.std) # (2)
    return (temps - einops.repeat(temp_avg, "a -> (a 7)")) / einops.repeat(temp_std, "a -> (a 7)")

tests.test_temperatures_normalized(temperatures_normalized)

def normalize_rows(matrix: Tensor) -> Tensor:
    """Normalize each row of the given 2D matrix.
    matrix: a 2D tensor of shape (m, n)

    Returns: a tensor of the same shape where row is divided by its l2 norm
    """
    return matrix / t.norm(matrix, p =2.0, dim=1, keepdim=True)

def normalize_rows_einops(matrix: Tensor) -> Tensor:
    """Normalize each row of the given 2D matrix.
    matrix: a 2D tensor of shape (m, n)

    Returns: a tensor of the same shape where row is divided by its l2 norm
    """
    return matrix / (einops.reduce(matrix**2, "a b -> a 1", "sum")**0.5)

tests.test_normalize_rows(normalize_rows)
tests.test_normalize_rows(normalize_rows_einops)

def cos_sim_matrix(matrix: Tensor) -> Tensor:
    """Return the cosine similarity matrix for each pair of rows of the given matrix.

    matrix: shape (m, n)
    """
    norm_rows =  matrix.norm(dim=1, p=2.0, keepdim=True) # (n, 1)
    return t.matmul(matrix, matrix.T) / t.matmul(norm_rows, norm_rows.reshape(1, -1))

tests.test_cos_sim_matrix(cos_sim_matrix)


def sample_distribution(probs: Tensor, n: int) -> Tensor:
    """Return n random samples from probs, where probs is a normalized probability distribution.

    probs: shape (k,) where probs[i] is the probability of event i occurring.
    n: number of random samples

    Return: shape (n,) where out[i] is an integer indicating which event was sampled.

    Use t.rand and t.cumsum to do this without any explicit loops.
    """
    # Oh i lack these 2 but i love these assertion
    assert abs(probs.sum() - 1) < 0.0001
    assert (probs >= 0).all()
    return t.sum(t.rand(n).reshape(-1, 1) > t.cumsum(probs, dim=0).reshape(1, -1), dim=1)

tests.test_sample_distribution(sample_distribution)

def classifier_accuracy(scores: Tensor, true_classes: Tensor) -> Tensor:
    """Return the fraction of inputs for which the maximum score corresponds to the true class for that input.

    scores: shape (batch, n_classes). A higher score[b, i] means that the classifier thinks class i is more likely.
    true_classes: shape (batch, ). true_classes[b] is an integer from [0...n_classes).

    Use t.argmax.
    """
    return t.mean(t.argmax(scores, dim=-1) == true_classes)

tests.test_classifier_accuracy(classifier_accuracy)
# %%
x = t.rand(4)
print(x)
probs = t.tensor([0.1, 0.3, 0.6])
t.cumsum(probs, dim=0)
x.reshape(-1, 1) > t.cumsum(probs, dim=0).reshape(1,3)

# %%
