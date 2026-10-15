#%%
import os
import sys
from functools import partial
from pathlib import Path
from typing import Callable

import einops
import plotly.express as px
import plotly.graph_objects as go
import torch as t
from IPython.display import display
from ipywidgets import interact
from jaxtyping import Bool, Float
from torch import Tensor
from tqdm.auto import tqdm
# %%
chapter = "chapter0_fundamentals"
section = "part1_ray_tracing"
# This will be $HOME/ml-knowledge/ARENA_3.0_Exercises
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part1_ray_tracing.tests as tests
from part1_ray_tracing.utils import (
    render_lines_with_plotly,
    setup_widget_fig_ray,
    setup_widget_fig_triangle
)
from plotly_utils import imshow

MAIN = __name__ == "__main__"

#%%
def make_rays_1d(num_pixels: int, y_limit: float) -> Tensor:
    """
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is
        also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains
        (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    """
    out = t.zeros(num_pixels, 2, 3).float()
    out[:, 1, 0] = 1.0
    out[:, 1, 1] = t.linspace(-y_limit, y_limit, num_pixels)
    return out

tests.test_make_rays_1d(make_rays_1d)
rays1d = make_rays_1d(9, 10.0)
fig = render_lines_with_plotly(rays1d)

#%%
fig: go.FigureWidget = setup_widget_fig_ray()
display(fig)


@interact(v=(0.0, 6.0, 0.01), seed=(0, 10, 1))
def update(v=0.0, seed=0):
    t.manual_seed(seed)
    L_1, L_2 = t.rand(2, 2)
    P = lambda v: L_1 + v * (L_2 - L_1)
    x, y = zip(P(0), P(6))
    with fig.batch_update():
        fig.update_traces({"x": x, "y": y}, 0)
        fig.update_traces({"x": [L_1[0], L_2[0]], "y": [L_1[1], L_2[1]]}, 1)
        fig.update_traces({"x": [P(v)[0]], "y": [P(v)[1]]}, 2)

#%%
def intersect_ray_1d(ray: Float[Tensor, "points dim"], segment: Float[Tensor, "points dims"]) -> bool:
    """
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    """
    A = t.zeros(2, 2, dtype=t.float32)
    A[:, 0] = ray[1][0:2]
    A[:, 1] = (segment[0] - segment[1])[0:2]
    b = (segment[0] - ray[0])[0:2] # it should be (2,)

    sol = t.zeros(2, dtype=t.float32)
    try:
        sol = t.linalg.solve(A, b, out = sol) # Can check singular without using .det
    except RuntimeError:
        return False

    if sol[0].item() >= 0 and (0 <= sol[1].item() <= 1):
        return True
    return False

tests.test_intersect_ray_1d(intersect_ray_1d)

#%%
def intersect_rays_1d(
        rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]
) -> Bool[Tensor, "nrays"]:
    """
    For each ray, return True if it intersects any segment 

    rays: [ray, O, D]
    segments : [segment, L1, L2] Cheston Tan
    """
    nrays = rays.shape[0] # can use rays.size(0)
    nsegments = segments.shape[0] # can use segments.size(0)

    # Ray [O, D]
    O, D = rays[:, 0, 0:2], rays[:, 1, 0:2] # [nrays, 2]

    # Segment [L1, L2]
    L_1, L_2 = segments[:, 0, 0:2], segments[:, 1, 0:2] # [nsegment, 2]

    # I think we need (nrays nsegment 2 2)
    l1subl2 = L_1 - L_2
    assert l1subl2.shape == (nsegments, 2)

    # A got shape [nrays nsegments 2 2]
    A = t.stack([einops.repeat(D, "nrays dim -> nrays nsegments dim", nsegments = nsegments), \
                einops.repeat(l1subl2, "nsegments dim -> nrays nsegments dim", nrays=nrays)], dim=-1)
    # b got shape [nrays nsegments 2]
    b = einops.rearrange(L_1, "nsegment dim -> 1 nsegment dim") - \
        einops.rearrange(O, "nrays dim -> nrays 1 dim")

    # Assign A's matrix to eye if det is 0
    mask = t.linalg.det(A).abs() < 1e-8 # [nrays nsegments]
    assert mask.size() == (nrays, nsegments)

    A[mask] = t.eye(2)

    # solve linear equation
    sol = t.linalg.solve(A, b) #[nrays nsegments 2]

    # Now we for each nray, we check if any segment that has u>=0, v>=0, v<=1
    # First set the sol's mask to be unsolutionalbe e.g. u=-1, v=-1
    sol[mask] = t.tensor([-1,-1]).float()
    # Now check valid solution 
    valid_sol_mask = (sol[..., 0] >= 0) & (sol[..., 1] >= 0) & (sol[..., 1] <= 1)
    # So valid_sol_mask is [nray, nsegment], now just check if each ray ANY is true
    return valid_sol_mask.any(dim=-1)

tests.test_intersect_rays_1d(intersect_rays_1d)

#%%
def make_rays_2d(num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float) -> Float[Tensor, "nrays 2 3"]:
    """
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    """
    yrays = t.linspace(-y_limit, y_limit, num_pixels_y)
    zrays = t.linspace(-z_limit, z_limit, num_pixels_z)

    # First is using the meshgrid
    grid_a, grid_b = t.meshgrid(yrays, zrays, indexing='ij')
    yz_coords = t.stack([grid_a, grid_b], dim=-1).reshape(-1, 2)

    A = t.zeros((num_pixels_z*num_pixels_y,2,3))
    A[:, 1, -2:] = yz_coords
    A[:, 1, 0] = 1.0

    # Second is using repeat
    B = t.zeros((num_pixels_y*num_pixels_z,2,3))
    B[:, 1, 0] = 1.0
    B[:, 1, 1] = einops.repeat(yrays, "a -> (a z)", z = num_pixels_z)
    B[:, 1, 2] = einops.repeat(zrays, "a -> (y a)", y = num_pixels_y)

    return B

tests.test_make_rays_2d(make_rays_2d)

#%%

rays = make_rays_2d(9, 9, 1, 1)
fig = render_lines_with_plotly(rays)

# %%

x = t.arange(3)
einops.repeat(x, "z -> (y z)", y =3)
# %%
