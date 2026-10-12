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
    return (t.argmax(scores, dim=-1) == true_classes).float().mean()

tests.test_classifier_accuracy(classifier_accuracy)

def total_price_indexing(prices: Tensor, items: Tensor) -> float:
    """Given prices for each kind of item and a tensor of items purchased, return the total price.

    prices: shape (k, ). prices[i] is the price of the ith item.
    items: shape (n, ). A 1D tensor where each value is an item index from [0..k).

    Use integer array indexing. The below document describes this for NumPy but it's the same in PyTorch:

    https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing
    """
    assert items.max() < len(prices)
    return prices[items].sum().item()

tests.test_total_price_indexing(total_price_indexing)

def gather_2d(matrix: Tensor, indexes: Tensor) -> Tensor:
    """Perform a gather operation along the second dimension.

    matrix: shape (m, n)
    indexes: shape (m, k)

    Return: shape (m, k). out[i][j] = matrix[i][indexes[i][j]]

    For this problem, the test already passes and it's your job to write at least three asserts relating the arguments and the output. This is a tricky function and worth spending some time to wrap your head around its behavior.

    See: https://pytorch.org/docs/stable/generated/torch.gather.html?highlight=gather#torch.gather
    """
    ## COde here - add assert statements here for `indices` and `matrix`
    #assert indexes.dim() == matrix.dim()
    assert indexes.ndim == matrix.ndim
    assert indexes.size(0) <= matrix.size(0)

    out = matrix.gather(1, indexes)
    # Code here - add assert statements here for `out`
    assert out.shape == indexes.shape

    return out

tests.test_gather_2d(gather_2d)

def total_price_gather(prices: Tensor, items: Tensor) -> float:
    """Compute the same as total_price_indexing, but use torch.gather"""
    assert items.max() < prices.shape[0]
    #return prices.gather(0, items).sum().item() # This pass
    return t.gather(prices, dim=0, index=items).sum().item()

tests.test_total_price_gather(total_price_gather)

def integer_array_indexing(matrix: Tensor, coords: Tensor) -> Tensor:
    """Return the values at each coordinate using integer array indexing.

    For details on integer array indexing, see:
    https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing

    matrix: shape (d_0, d_1, ..., d_n)
    coords: shape (batch, n)

    Return: (batch, )
    """
    return matrix[tuple(coords.T)]

tests.test_integer_array_indexing(integer_array_indexing)

def batched_logsumexp(matrix: Tensor) -> Tensor:
    """For each row of the matrix, compute log(sum(exp(row))) in a numerically stable way.

    matrix: shape (batch, n)

    Return: (batch, ). For each i, out[i] = log(sum(exp(matrix[i]))).

    Do this without using PyTorch's logsumexp function.

    A couple useful blogs about this function:
    - https://leimao.github.io/blog/LogSumExp/
    - https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
    """
    #return (matrix - matrix.max(dim=-1).values[:, None]).exp().sum(dim=1).log() + matrix.max(dim=-1).values # passed one
    C = matrix.max(dim=-1).values
    exp = t.exp(matrix - einops.rearrange(C, "a -> a 1"))
    sum_exp = t.sum(exp, dim=-1)
    log_exp = t.log(sum_exp)
    return C + log_exp

tests.test_batched_logsumexp(batched_logsumexp)

def batched_softmax(matrix: Tensor) -> Tensor:
    """For each row of the matrix, compute softmax(row).

    Do this without using PyTorch's softmax function.
    Instead, use the definition of softmax: https://en.wikipedia.org/wiki/Softmax_function

    matrix: shape (batch, n)

    Return: (batch, n). For each i, out[i] should sum to 1.
    """
    # wait what if i use directly exp should be exponentially large/small?
    # this somehow passed
    ### HAHA okie according to solution, this is the solution cuz it does not care exp stability now
    sum_exp = matrix.exp().sum(dim=-1)
    return matrix.exp() / einops.rearrange(sum_exp, "a -> a 1")

    # how about using the logsumexp? it works
    #logsumexp = batched_logsumexp(matrix)
    #return (matrix - einops.rearrange(logsumexp, "a -> a 1")).exp()

    # somehow Wikipedia just saying to use the max value for each row? let's try
    # okie it works
    C = matrix.max(dim=-1).values
    sum_exp = (matrix - einops.rearrange(C, "t -> t 1")).exp().sum(dim=-1)
    return (matrix - einops.rearrange(C, "t -> t 1")).exp() / einops.rearrange(sum_exp, "t -> t 1")

tests.test_batched_softmax(batched_softmax)

def batched_logsoftmax(matrix: Tensor) -> Tensor:
    """Compute log(softmax(row)) for each row of the matrix.

    matrix: shape (batch, n)

    Return: (batch, n).

    Do this without using PyTorch's logsoftmax function.
    For each row, subtract the maximum first to avoid overflow if the row contains large values.
    """
    # i use this version but not working and it is inf/nan
    #logsumexp = batched_logsumexp(matrix)
    #return (matrix - einops.rearrange(logsumexp, "t -> t 1")).exp().log()
    
    # this works
    #logsumexp = batched_logsumexp(matrix)
    #return matrix - einops.rearrange(logsumexp, "t -> t 1")

    # let's try the max value subtract. but this not work either
    #max_value = einops.rearrange(matrix.max(dim=-1).values, "t -> t 1")
    #sum_exp = (matrix - max_value).exp().sum(dim=-1)
    #softmax = (matrix - max_value).exp() / einops.rearrange(sum_exp, "t -> t 1")
    #return softmax.log()

    # let's try just the numerical stability. this work
    max_value = einops.rearrange(matrix.max(dim=-1).values, "t -> t 1")
    sum_exp = (matrix - max_value).exp().sum(dim=-1)
    return (matrix - max_value) - einops.rearrange(sum_exp.log(), "t -> t 1")

tests.test_batched_logsoftmax(batched_logsoftmax)

def batched_cross_entropy_loss(logits: Tensor, true_labels: Tensor) -> Tensor:
    """Compute the cross entropy loss for each example in the batch.

    logits: shape (batch, classes). logits[i][j] is the unnormalized prediction for example i and class j.
    true_labels: shape (batch, ). true_labels[i] is an integer index representing the true class for example i.

    Return: shape (batch, ). out[i] is the loss for example i.

    Hint: convert the logits to log-probabilities using your batched_logsoftmax from above.
    Then the loss for an example is just the negative of the log-probability that the model assigned to the true class. Use torch.gather to perform the indexing.
    """
    logsoftmax = batched_logsoftmax(logits)
    return einops.rearrange(-logsoftmax.gather(dim=1, index=true_labels[:, None]), "t 1 -> t")

tests.test_batched_cross_entropy_loss(batched_cross_entropy_loss)

#%%
matrix = t.arange(1,16).reshape(5,3)
print(matrix)
indices = t.tensor([1,1,1,1,1]).reshape(5,1)
t.gather(matrix, dim=1, index=indices)

# %% for gathering
matrix = t.arange(1, 9).reshape(2,4)
print(matrix)
print(matrix.shape)
print(matrix.size())
#indexes = t.tensor([[1, 1, 1, 1, 1], [1, 0, 0, 0, 1]]) # invalid size cuz gather dim 0 so dim 1 cannot bigger because output's dim 1 get value from matrix' dim 1
indexes = t.tensor([[1, 1, 1, 1], [1, 0, 0, 0], [0,0,0,0]]) # still valid cuz gather dim 0 so dim 0 can bigger than original matrix size
matrix.gather(0, indexes)

# %%
