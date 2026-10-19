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
one_triangle = t.tensor([[0, 0, 0], [4, 0.5, 0], [2, 3, 0]])
A, B, C = one_triangle
x, y, z = one_triangle.T

fig: go.FigureWidget = setup_widget_fig_triangle(x, y, z)
display(fig)

@interact(u=(-0.5, 1.5, 0.01), v=(-0.5, 1.5, 0.01))
def update(u=0.0, v=0.0):
    P = A + u * (B - A) + v * (C - A)
    fig.update_traces({"x": [P[0]], "y": [P[1]]}, 2)

# %%
Point = Float[Tensor, "points=3"]

def triangle_ray_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    """
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the ray and the triangle intersect.
    """
    mat = t.stack([-D, B-A, C-A], dim=-1)
    b = O - A

    try:
        sol = t.linalg.solve(mat, b)
    except:
        return False

    s, u, v = sol

    return ((s >= 0) & (0 <= u) & (0 <= v) & (u + v <= 1)).item()

tests.test_triangle_ray_intersects(triangle_ray_intersects)

# %%
def raytrace_triangle(
        rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
        triangle: Float[Tensor, "trianglePoints=3 dims=3"],) -> Bool[Tensor, "nrays"]:
    """
    For each ray, return True if the triangle intersects that ray 
    """
    # Extract A, B, C, D, O
    A, B, C = triangle[0], triangle[1], triangle[2]
    assert A.shape == (3,)

    O, D = t.unbind(rays, dim=1)
    nrays = rays.shape[0]
    assert D.shape == (nrays, 3)

    # First, then the mat and b for solving equation
    mat = t.stack([-D, einops.repeat(B-A, "a -> rays a", rays=nrays),
                    einops.repeat(C-A, "a -> rays a", rays=nrays)], dim=-1)
    assert mat.shape == (nrays, 3, 3)

    b = O - A
    assert b.shape == (nrays, 3)

    # Find singular matrix
    is_singular = t.linalg.det(mat).abs() < 1e-6
    assert is_singular.shape == (nrays,)
    # Then replace
    mat[is_singular] = t.eye(3)

    # Now solving
    sol = t.linalg.solve(mat, b)
    assert sol.shape == (nrays, 3)

    s, u, v = sol[..., 0], sol[..., 1], sol[..., 2]

    return (0 <= u) & (0 <= v) & (u + v <= 1) & (s >= 0) & (~is_singular)

tests.test_raytrace_triangle(raytrace_triangle)

# %%

A = t.tensor([1, 0.0, -0.5])
B = t.tensor([1, -0.5, 0.0])
C = t.tensor([1, 0.5, 0.5])
num_pixels_y = num_pixels_z = 15
y_limit = z_limit = 0.5

# Plot triangles and rays
test_triangle = t.stack([A, B, C], dim=0)
rays2d = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
triangle_lines = t.stack([A, B, C, A, B, C], dim=0).reshape(-1, 2, 3)
render_lines_with_plotly(rays2d, triangle_lines)

# Calculate and display intersections
intersects = raytrace_triangle(rays2d, test_triangle)
img = intersects.reshape(num_pixels_y, num_pixels_z).int()
imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")

# %%
triangles = t.load(section_dir / "pikachu.pt", weights_only=True)

def raytrace_mesh(
        rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
        triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"]
) -> Float[Tensor, "nrays"]:
    """
    For each ray, return the distance to the closest intersecting triangle, or infinity 
    """

    nrays = rays.shape[0]
    ntriangles = triangles.shape[0]

    # Step 1 is to extract A, B, C, O, D
    A, B, C = triangles.unbind(dim=1)
    assert A.shape == (ntriangles, 3)

    O, D = rays.unbind(dim=1)
    assert O.shape == (nrays, 3)

    # Step 2 is to repeat the shape so each A, B, C, O, D should have (nrays ntriangles, 3)
    A = einops.repeat(A, "ntriangles dims -> nrays ntriangles dims", nrays = nrays)
    B = einops.repeat(B, "ntriangles dims -> nrays ntriangles dims", nrays = nrays)
    C = einops.repeat(C, "ntriangles dims -> nrays ntriangles dims", nrays = nrays)
    O = einops.repeat(O, "nrays dims -> nrays ntriangles dims", ntriangles=ntriangles)
    D = einops.repeat(D, "nrays dims -> nrays ntriangles dims", ntriangles=ntriangles)

    # Step 3 is to construct mat and vec
    mat = t.stack([-D, B-A, C-A], dim=-1)
    assert mat.shape == (nrays, ntriangles, 3, 3)
    is_singular = t.linalg.det(mat).abs() < 1e-6
    assert is_singular.shape == (nrays, ntriangles)

    mat[is_singular] = t.eye(3)

    vec = O - A
    assert vec.shape == (nrays, ntriangles, 3)

    # Step 4 is to solve
    sol = t.linalg.solve(mat, vec)
    assert sol.shape == (nrays, ntriangles, 3)

    s, u, v = sol.unbind(dim=-1) # should have shape (nrays, ntriangles)
    assert s.shape == (nrays, ntriangles)
    # This is valid for both nrays, ntriangles
    valid_mask = (0 <= u) & (0 <= v) & (u + v <= 1) & (s >= 0) & (~is_singular)
    assert valid_mask.shape == (nrays, ntriangles)
    # But because we only care about ray so we will reduce last dim with any triangles intersect with each ray
    valid_mask = valid_mask.any(dim=-1)
    assert valid_mask.shape == (nrays,)

    # Step 5 is to find the distance which is O + sD
    ray_equation = O + t.mul(s[..., None], D) # must use mul, not matmul
    all_dist = einops.reduce(ray_equation, "nrays ntriangles dim -> nrays ntriangles",
                             lambda x, dim: (x**2).sum(dim=dim).sqrt())
    min_dist = einops.reduce(all_dist, "nrays ntriangles -> nrays", "min")

    min_dist[~valid_mask] = float('inf')
    return min_dist

num_pixels_y=120
num_pixels_z=120
y_limit = z_limit = 1

rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
rays[:, 0] = t.tensor([-2, 0.0, 0.0])
dists = raytrace_mesh(rays, triangles)
intersects = t.isfinite(dists).view(num_pixels_y, num_pixels_z)
dists_square = dists.view(num_pixels_y, num_pixels_z)
img = t.stack([intersects, dists_square], dim=0)

fig = px.imshow(img, facet_col=0, origin="lower", color_continuous_scale="magma", width=1000)
fig.update_layout(coloraxis_showscale=False)
for i, text in enumerate(["Intersects, Distance"]):
    fig.layout.annotations[i]["text"] = text
fig.show()

#%%
def rotation_matrix(theta: Float[Tensor, ""]) -> Float[Tensor, "rows cols"]:
    """
    Creates a rotation matrix representing a counterclockwise rotation of `theta` (radian) around the y-axis 
    """
    rad = theta
    return t.tensor([[t.cos(rad), 0, t.sin(rad)], [0, 1, 0], [-t.sin(rad), 0, t.cos(rad)]])

tests.test_rotation_matrix(rotation_matrix)

def raytrace_mesh_video(
        rays: Float[Tensor, "nrays points dim"],
        triangles: Float[Tensor, "ntriangles points dims"],
        rotation_matrix: Callable[[float], Float[Tensor, "rows cols"]],
        raytrace_function: Callable,
        num_frames: int,
) -> Bool[Tensor, "nframes nrays"]:
    """
    Creates a stack of raytracing results, rotating the triangles by `rotation_matrix` each frame 
    """
    result = []
    theta = t.tensor(2 * t.pi) / num_frames
    R = rotation_matrix(theta)
    for _ in tqdm(range(num_frames)):
        triangles = triangles @ R
        result.append(raytrace_function(rays, triangles))
        t.cuda.empty_cache() # clear GPU memory (this line will be more important later on)

    return t.stack(result, dim=0)

def display_video(distances: Float[Tensor, "frames y z"]):
    """
    Display video of raytracing results, using Plotly. `distances` is a tensor where the [i,y,z]  element
    is distance to the closest triangle for the i-th frame & the [y, z]-th ray in our 2D
    grid of rays
    """
    px.imshow(
        distances,
        animation_frame=0,
        origin="lower",
        zmin=0.0,
        zmax=distances[distances.isfinite()].quantile(0.99).item(),
        color_continuous_scale="viridis_r",  # "Brwnyl"
    ).update_layout(coloraxis_showscale=False, width=550, height=600, title="Raytrace mesh video").show()

num_pixels_y = 250
num_pixels_z = 250
y_limit = z_limit = 0.8
num_frames = 50

rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
rays[:, 0] = t.tensor([-3.0, 0.0, 0.0])
dists = raytrace_mesh_video(rays, triangles, rotation_matrix, raytrace_mesh, num_frames)
dists = einops.rearrange(dists, "frames (y z) -> frames y z", y = num_pixels_y)

display_video(dists)
# %%
print(t.cuda.is_available())
print(t.backends.mps.is_available())
# %%
def raytrace_mesh_gpu(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"],
) -> Float[Tensor, " nrays"]:
    """
    For each ray, return the distance to the closest intersecting triangle, or infinity.

    All computations should be performed on the GPU
    """
    if t.cuda.is_available():
        device = t.device("cuda")
    elif t.backends.mps.is_available():
        device = t.device("mps")
    else:
        device = t.device("cpu")

    rays = rays.to(device)
    triangles = triangles.to(device)

    nrays = rays.shape[0]
    ntriangles = triangles.shape[0]

    # Step 1 is to extract A, B, C, O, D
    A, B, C = triangles.unbind(dim=1)
    assert A.shape == (ntriangles, 3)

    O, D = rays.unbind(dim=1)
    assert O.shape == (nrays, 3)

    # Step 2 is to repeat the shape so each A, B, C, O, D should have (nrays ntriangles, 3)
    A = einops.repeat(A, "ntriangles dims -> nrays ntriangles dims", nrays = nrays)
    B = einops.repeat(B, "ntriangles dims -> nrays ntriangles dims", nrays = nrays)
    C = einops.repeat(C, "ntriangles dims -> nrays ntriangles dims", nrays = nrays)
    O = einops.repeat(O, "nrays dims -> nrays ntriangles dims", ntriangles=ntriangles)
    D = einops.repeat(D, "nrays dims -> nrays ntriangles dims", ntriangles=ntriangles)

    # Step 3 is to construct mat and vec
    mat = t.stack([-D, B-A, C-A], dim=-1)
    assert mat.shape == (nrays, ntriangles, 3, 3)
    is_singular = t.linalg.det(mat).abs() < 1e-6
    assert is_singular.shape == (nrays, ntriangles)

    mat[is_singular] = t.eye(3).to(device)

    vec = O - A
    assert vec.shape == (nrays, ntriangles, 3)

    # Step 4 is to solve
    sol = t.linalg.solve(mat, vec)
    assert sol.shape == (nrays, ntriangles, 3)

    s, u, v = sol.unbind(dim=-1) # should have shape (nrays, ntriangles)
    assert s.shape == (nrays, ntriangles)
    s = s * D[..., 0]
    # This is valid for both nrays, ntriangles
    valid_mask = (0 <= u) & (0 <= v) & (u + v <= 1) & (s >= 0) & (~is_singular)
    assert valid_mask.shape == (nrays, ntriangles)
    # But because we only care about ray so we will reduce last dim with any triangles intersect with each ray
    #valid_mask = valid_mask.any(dim=-1)
    #assert valid_mask.shape == (nrays,)

    # Step 5 is to find the distance which is O + sD
    ray_equation = O + t.mul(s[..., None], D) # must use mul, not matmul
    all_dist = einops.reduce(ray_equation, "nrays ntriangles dim -> nrays ntriangles",
                             lambda x, dim: (x**2).sum(dim=dim).sqrt())
    all_dist[~valid_mask] = t.inf
    min_dist = einops.reduce(all_dist, "nrays ntriangles -> nrays", "min")

    # These lines of code give output as expected as solutions
    #s[~valid_mask] = t.inf
    #min_dist = einops.reduce(s, "nr nt -> nr", "min")

    return min_dist.to("cpu")

dists = raytrace_mesh_video(rays, triangles, rotation_matrix, raytrace_mesh_gpu, num_frames)
dists = einops.rearrange(dists, "frames (y z) -> frames y z", y=num_pixels_y)
display_video(dists)
# %%
device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

def raytrace_mesh_lambert(
        rays: Float[Tensor, "nrays points=2 dims=3"],
        triangles: Float[Tensor, "ntriangles points=3 dims=3"],
        light: Float[Tensor, "dims=3"],
        ambient_intensity: float,
        device: str = device,
) -> Float[Tensor, "nrays"]:
    """
    For each ray, return the intensity of light hitting the triangle it intersects with (or zero if
    no intersection).

    Args:
        rays:   A tensor of rays, with shape `[nrays, 2, 3]`.
        triangles:  A tensor of triangles, with shape `[ntriangles, 3, 3]`.
        light:  A tensor representing the light vector, with shape `[3]`. We compute the intensity
                as the dot product of the triangle normals & the light vector, then set it to be
                zero if the sign is negative.
        ambient_intensity:  A float representing the ambient intensity. This is the minimum
                            brightness for a triangle, to differentiate it from the black background
                            (rays that don't hit any triangle).
        device: The device to perform the computation on.

    Returns:
        A tensor of intensities for each of the rays, flattened over the [y, z] dimensions. The
        values are zero when there is no intersection, and `ambient_intensity + intensity` when
        there is an intersection (where `intensity` is the dot product of the triangle's normal
        vector and the light vector, truncated at zero).
    """
    rays = rays.to(device)
    triangles = triangles.to(device)
    light = light.to(device)

    nrays = rays.shape[0]
    ntriangles = triangles.shape[0]

    # Step 1 is to extract A, B, C, O, D
    A, B, C = triangles.unbind(dim=1)
    assert A.shape == (ntriangles, 3)

    O, D = rays.unbind(dim=1)
    assert O.shape == (nrays, 3)

    # Step 2 is to repeat the shape so each A, B, C, O, D should have (nrays ntriangles, 3)
    A = einops.repeat(A, "ntriangles dims -> nrays ntriangles dims", nrays = nrays)
    B = einops.repeat(B, "ntriangles dims -> nrays ntriangles dims", nrays = nrays)
    C = einops.repeat(C, "ntriangles dims -> nrays ntriangles dims", nrays = nrays)
    O = einops.repeat(O, "nrays dims -> nrays ntriangles dims", ntriangles=ntriangles)
    D = einops.repeat(D, "nrays dims -> nrays ntriangles dims", ntriangles=ntriangles)

    # Step 3 is to construct mat and vec
    mat = t.stack([-D, B-A, C-A], dim=-1)
    assert mat.shape == (nrays, ntriangles, 3, 3)
    is_singular = t.linalg.det(mat).abs() < 1e-6
    assert is_singular.shape == (nrays, ntriangles)

    mat[is_singular] = t.eye(3).to(device)

    vec = O - A
    assert vec.shape == (nrays, ntriangles, 3)

    # Step 4 is to solve
    sol = t.linalg.solve(mat, vec)
    assert sol.shape == (nrays, ntriangles, 3)

    s, u, v = sol.unbind(dim=-1) # should have shape (nrays, ntriangles)
    assert s.shape == (nrays, ntriangles)
    # This is valid for both nrays, ntriangles
    s = s * D[..., 0]
    valid_mask = (0 <= u) & (0 <= v) & (u + v <= 1) & (s >= 0) & (~is_singular)
    assert valid_mask.shape == (nrays, ntriangles)
    ## But because we only care about ray so we will reduce last dim with any triangles intersect with each ray
    # -> THIS IS THE CULPRIT for the wrong ray tracing, should not reduce last dim but let it be there and
    #valid_mask = valid_mask.any(dim=-1)
    #assert valid_mask.shape == (nrays,)

    # Step 5 is to find the distance which is O + sD and its min position
    ray_equation = O + t.mul(s[..., None], D) # must use mul, not matmul
    all_dist = einops.reduce(ray_equation, "nrays ntriangles dim -> nrays ntriangles",
                             lambda x, dim: (x**2).sum(dim=dim).sqrt())

    all_dist[~valid_mask] = t.inf
    closest_distances, closest_triangles_for_each_ray = t.min(all_dist, dim=-1)
    assert closest_distances.shape == (nrays,)
    assert closest_triangles_for_each_ray.shape == (nrays,)

    # Step 6 is to find normal vector of all triangles
    v1 = A - B
    v2 = C - B
    norm_vector: Tensor[Float, "nrays ntriangles 3"] = t.stack([t.mul(v1[..., 1], v2[..., 2]) - t.mul(v1[..., 2], v2[..., 1]),
                                                                t.mul(v1[..., 2], v2[..., 0]) - t.mul(v1[..., 0], v2[..., 2]),
                                                                t.mul(v1[..., 0], v2[..., 1]) - t.mul(v1[..., 1], v2[..., 0])], dim=-1)
    norm_vector = norm_vector / norm_vector.norm(dim=2, keepdim=True)
    assert norm_vector.shape == (nrays, ntriangles, 3)

    # Step 7 is dot product, then from min_dist find the initensity of min_dist_index
    intensity_per_triangle = einops.einsum(norm_vector, light, "i j k, k -> i j")
    intensity_per_triangle_signed = t.where(intensity_per_triangle > 0, intensity_per_triangle, 0.0) # this signed check if light is coming from the wrong side of triangle
    assert intensity_per_triangle_signed.shape == (nrays, ntriangles)

    intensity = intensity_per_triangle_signed.gather(dim=-1, index=closest_triangles_for_each_ray.reshape(nrays, 1)).squeeze() + ambient_intensity
    assert intensity.shape == (nrays,)

    intensity = t.where(closest_distances.isfinite(), intensity, 0)

    return intensity.cpu()

def display_video_with_lighting(intensity: Float[Tensor, "frames y z"]):
    """
    Displays video of raytracing results, using Plotly. `distances` is a tensor where the [i, y, z]
    element is the lighting intensity based on the angle of light & the surface of the triangle
    which this ray hits first.
    """
    px.imshow(
        intensity,
        animation_frame=0,
        origin="lower",
        color_continuous_scale="magma",
    ).update_layout(coloraxis_showscale=False, width=550, height=600, title="Raytrace mesh video (lighting)").show()

ambient_intensity = 0.5
light = t.tensor([0.0, -1.0, 1.0])
raytrace_function = partial(
    raytrace_mesh_lambert,
    ambient_intensity=ambient_intensity,
    light=light,
)

intensity = raytrace_mesh_video(rays, triangles, rotation_matrix, raytrace_function, num_frames)
intensity = einops.rearrange(intensity, "frames (y z) -> frames y z", y=num_pixels_y)
display_video_with_lighting(intensity)
# %%
v = t.arange(36).reshape(3,4,3)
print(v)
temp = t.tensor([1,1,1])
hihi = einops.einsum(v, temp, "i j k, k -> i j") # 3, 4
print(hihi)
min_index = t.tensor([1,2,3])
t.gather(hihi, dim=1, index=min_index.reshape(3,1))
# %%
# Before intensity is [ntrianges,]
intensity = t.arange(5, 5+12).reshape(12)
print(intensity)
min_index = t.tensor([0, 0, 1, 5]) # nr
print(intensity[min_index])

# After intensity is [nray, ntrianges]
intensity = t.arange(4*12).reshape(4, 12) # 4 rays, each with 12 triangles
print(intensity)
t.gather(intensity, dim=1, index=min_index.reshape(4,1))
# %%
def raytrace_triangle_with_bug(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"]
) -> Bool[Tensor, " nrays"]:
    '''
    For each ray, return True if the triangle intersects that ray.
    '''
    NR = rays.size(0)
    print(NR)

    A, B, C = einops.repeat(triangle, "pts dims -> pts NR dims", NR=NR)

    O, D = rays.unbind(1)

    mat = t.stack([- D, B - A, C - A], dim=-1)

    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-6
    mat[is_singular] = t.eye(3)

    vec = O - A

    sol = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(dim=-1)

    return ((u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular & (s >= 0))


intersects = raytrace_triangle_with_bug(rays2d, test_triangle)
img = intersects.reshape(num_pixels_y, num_pixels_z).int()
imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")

# %%
