#%%
import os
import sys
import torch as t
from torch import Tensor
import torch
import einops
from ipywidgets import interact
import plotly.express as px
from ipywidgets import interact
from pathlib import Path
from IPython.display import display
from jaxtyping import Float, Int, Bool, Shaped, jaxtyped
import typeguard

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part1_ray_tracing"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow
from part1_ray_tracing.utils import render_lines_with_plotly, setup_widget_fig_ray, setup_widget_fig_triangle
import part1_ray_tracing.tests as tests

MAIN = __name__ == "__main__"

#%%
def make_rays_1d(num_pixels: int, y_limit: float) -> t.Tensor:
    '''
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) 
    where the num_points dimension contains (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    '''
    rays = torch.zeros((num_pixels, 2, 3), dtype=torch.float32)
    rays[:, 1, 0] = 1
    rays[:, 1, 1] = torch.linspace(-y_limit, y_limit, steps = num_pixels)
    return rays

rays1d = make_rays_1d(9, 10.0)

if MAIN:
    fig = render_lines_with_plotly(rays1d)

#%%

if MAIN:
    fig = setup_widget_fig_ray()
    display(fig)

@interact
def response(seed=(0, 10, 1), v=(-2.0, 2.0, 0.01)):
    t.manual_seed(seed)
    L_1, L_2 = t.rand(2, 2)
    P = lambda v: L_1 + v * (L_2 - L_1)
    x, y = zip(P(-2), P(2))
    with fig.batch_update(): 
        fig.data[0].update({"x": x, "y": y}) 
        fig.data[1].update({"x": [L_1[0], L_2[0]], "y": [L_1[1], L_2[1]]}) 
        fig.data[2].update({"x": [P(v)[0]], "y": [P(v)[1]]})
# %%
segments = t.tensor([
    [[1.0, -12.0, 0.0], [1, -6.0, 0.0]], 
    [[0.5, 0.1, 0.0], [0.5, 1.15, 0.0]], 
    [[2, 12.0, 0.0], [2, 21.0, 0.0]]
])

if MAIN:
    fig = render_lines_with_plotly(rays1d, segments)
# %%
def intersect_ray_1d(ray: t.Tensor, segment: t.Tensor) -> bool:
    '''
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    '''
    #vec2 = segments[1] - segments[0]
    #print(segments)
    #print(vec2.shape)
    #c = torch.cross(vec1, vec2, dim=0)
    #if abs(torch.prod(c)) <= 1e-9:
        #return False

    Dxy = ray[1, :2] # (,2)
    L1_L2xy = segment[0, :2] - segment[1, :2] # (2,) - (2,) -> (2,)

    L1_Oxy = segment[0, :2] - ray[0, :2] #(2,) - (2,) -> (2,)
    # Defined Ax = b
    A = torch.stack([Dxy, L1_L2xy], dim=-1) # Dim must be -1 otherwise wrong.
    # I mistake here as I think because input is (n,) then either dim = 0 or -1 be fine. but not
    b = L1_Oxy
    try:
        x = torch.linalg.solve(A, b)
    except:
        return False
    
    u = x[0].item() # convert from torch scalar to python scalar
    v = x[1].item() # convert from torch scalar to python scalar
    if u < 0 or (v <0 or v > 1):
        return False
    return True

tests.test_intersect_ray_1d(intersect_ray_1d)
tests.test_intersect_ray_1d_special_case(intersect_ray_1d)
# %%

@jaxtyped
@typeguard.typechecked
def intersect_ray_1d(ray: Float[Tensor, "n=2 d=3"], segment: Float[Tensor, "n=2 d=3"]) -> bool:
    '''
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    '''
    #vec2 = segments[1] - segments[0]
    #print(segments)
    #print(vec2.shape)
    #c = torch.cross(vec1, vec2, dim=0)
    #if abs(torch.prod(c)) <= 1e-9:
        #return False

    Dxy = ray[1, :2] # (,2)
    L1_L2xy = segment[0, :2] - segment[1, :2] # (2,) - (2,) -> (2,)

    L1_Oxy = segment[0, :2] - ray[0, :2] #(2,) - (2,) -> (2,)
    # Defined Ax = b
    A = torch.stack([Dxy, L1_L2xy], dim=-1) # Dim must be -1 otherwise wrong.
    # I mistake here as I think because input is (n,) then either dim = 0 or -1 be fine. but not
    b = L1_Oxy
    try:
        x = torch.linalg.solve(A, b)
    except:
        return False
    
    u = x[0].item() # convert from torch scalar to python scalar
    v = x[1].item() # convert from torch scalar to python scalar
    if u < 0 or (v <0 or v > 1):
        return False
    return True

tests.test_intersect_ray_1d(intersect_ray_1d)
tests.test_intersect_ray_1d_special_case(intersect_ray_1d)
# %%

def intersect_rays_1d(rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if it intersects any segment.
    '''
    # get xy coordinates
    rays = rays[..., :2]
    segments = segments[..., :2] 

    print(f"rays: {rays}")
    print(f"segments: {segments}")

    # Take O and D, L_1, L2
    O = rays[:, 0, :] # shape (nrays, 2)
    D = rays[:, 1, :] # shape (nrays, 2)
    L_1 = segments[:, 0, :]
    L_2 = segments[:, 1, :]


    L1_L2 = L_1 - L_2 # (nsegments, 2)

    print(f"D: {D}")
    print(f"L1: {L_1}")
    print(f"L2: {L_2}")
    print(f"L1 - L2 : {L1_L2}")

    # we do (nrays, nsegments, ...)
    # First is for L1 - O which should have shape (nrays, nsegments, 2)
    O_extended = einops.repeat(O, 'nrays a -> nrays e a', e = 1)
    L1_extended = einops.repeat(L_1, 'nsegments a -> e nsegments a', e=1)
    L1_O = L1_extended - O_extended
    # Second, we do concate [D, L1-L2] to have shape (nrays, nsegments, 2, 2)
    D_extended = einops.repeat(D, 'nrays a -> nrays nsegments a', nsegments = segments.size(0))
    L1_L2_extended = einops.repeat(L1_L2, 'nsegments a -> nrays nsegments a', nrays = rays.size(0))
    DL = torch.stack([D_extended, L1_L2_extended], dim=-1)

    print(f"DL: {DL}")

    # Find determinate
    det = torch.linalg.det(DL) # (nrays, nsegments) 
    print(f"Det: {det}")
    DL[abs(det) < 1e-6] = torch.eye(2)
    print(f"New DL after det {DL}")
    sol = torch.linalg.solve(DL, L1_O) # (nrays, nsegments, 2)
    print(f"sol: {sol}")

    temp = ((sol[..., 0] >= 0) & (sol[..., 1] >= 0) & (sol[..., 1] <= 1)) # (nrays, nsegments)
    print(f"temp: {temp}")
    # Now from det, we replace temp[i][j] = False if det[i][j] = false
    temp[abs(det) < 1e-6] = False
    print(f"after temp: {temp}")
    return torch.any(temp, dim=-1)



if MAIN:
    tests.test_intersect_rays_1d(intersect_rays_1d)
    tests.test_intersect_rays_1d_special_case(intersect_rays_1d)


# %%
def make_rays_2d(num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float) -> Float[t.Tensor, "nrays 2 3"]:
    '''
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    '''
    rays_y = torch.zeros((num_pixels_y,  2, 3), dtype=torch.float32)
    rays_y[:, 1, 0] = 1
    rays_y[:, 1, 2] = 1
    rays_y[:, 1, 1] = torch.linspace(-y_limit, y_limit, steps = num_pixels_y)
    rays_y = einops.repeat(rays_y, 'n p a -> (n 10) p a')

    rays_z = torch.zeros((num_pixels_z,  2, 3), dtype=torch.float32)
    rays_z[:, 1, 0] = 1
    rays_z[:, 1, 1] = 1
    rays_z[:, 1, 2] = torch.linspace(-z_limit, z_limit, steps = num_pixels_z)
    rays_z = einops.repeat(rays_z, 'n p a -> (10 n) p a')

    rays = einops.einsum(rays_y, rays_z, 'n a b, n a b -> n a b')

    ## second ways to build it from empty
    n_pixels = num_pixels_y * num_pixels_z
    ygrid = torch.linspace(-y_limit, y_limit, steps = num_pixels_y)
    zgrid = torch.linspace(-z_limit, z_limit, steps = num_pixels_z)
    rays2 = torch.zeros((n_pixels, 2, 3), dtype=torch.float32)
    rays2[:, 1, 0] = 1
    rays2[:, 1, 1] = einops.repeat(ygrid, 'z -> (z n)', n = num_pixels_y)
    rays2[:, 1, 2] = einops.repeat(zgrid, 'z -> (n z)', n = num_pixels_z)


    return rays2


if MAIN:
    rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
    render_lines_with_plotly(rays_2d)
# %%

a = torch.tensor([1,2,3])
print(einops.repeat(a, 'b -> (b 5)')) #[[1,1,1,1,1],[2,2,2,2,2],[3,3,3,3,3]]
print(einops.repeat(a, 'b -> (5 b)')) # [[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]]
# %%
if MAIN:
    one_triangle = t.tensor([[0, 0, 0], [3, 0.5, 0], [2, 3, 0]])
    A, B, C = one_triangle
    x, y, z = one_triangle.T

    fig = setup_widget_fig_triangle(x, y, z)

@interact(u=(-0.5, 1.5, 0.01), v=(-0.5, 1.5, 0.01))
def response(u=0.0, v=0.0):
    P = A + u * (B - A) + v * (C - A)
    fig.data[2].update({"x": [P[0]], "y": [P[1]]})


if MAIN:
    display(fig)

# %%
Point = Float[Tensor, "points=3"]

@jaxtyped
@typeguard.typechecked
def triangle_ray_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    '''
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the ray and the triangle intersect.
    '''
    BA = B - A
    CA = C - A
    OA = O - A

    mat = torch.stack([-D, BA, CA], dim=-1)
    b = OA
    try:
        sol = torch.linalg.solve(mat, b)
    except:
        return False

    w, u, v = sol 
    return ((u >= 0) & (v >= 0) & (u + v <= 1)).item()


if MAIN:
    tests.test_triangle_ray_intersects(triangle_ray_intersects)
# %%



# %%
def raytrace_triangle(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"]
) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if the triangle intersects that ray.
    '''

    NR = rays.size(0)

    # Repeat rays and segments so that we can compuate the intersection of every (ray, segment) pair
    triangles = einops.repeat(triangle, "p d -> nrays p d", nrays=NR)

    # Each element of `rays` is [[Ox, Oy, Oz], [Dx, Dy, Dz]]
    O = rays[:, 0] # (nrays, 3)
    D = rays[:, 1] # (nrays, 3)
    assert O.shape == (NR, 3)
    assert D.shape == (NR, 3)

    # Each point A, B, C
    A = triangles[:, 0]
    B = triangles[:, 1]
    C = triangles[:, 2]

    assert A.shape == (NR, 3)
    assert B.shape == (NR, 3)
    assert C.shape == (NR, 3)

    # Define matrix on left hand side of equation
    mat = t.stack([-D, B - A , C-A], dim=-1)
    # Get boolean of where matrix is singular, and replace it with the identity in these positions
    dets = t.linalg.det(mat) # (NR,)
    is_singular = dets.abs() < 1e-8 #(NR,)
    assert is_singular.shape == (NR,)
    mat[is_singular] = t.eye(3)

    # Define vector on the right hand side of equation
    vec = O - A

    # Solve equation, get results
    sol = t.linalg.solve(mat, vec) #(NR, 3)
    s = sol[..., 0] #(NR,)
    u = sol[..., 1]
    v = sol[..., 2]

    # Return boolean of (matrix is nonsingular, and solution is in correct range implying intersection)
    return ((u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular)


if MAIN:
    A = t.tensor([1, 0.0, -0.5])
    B = t.tensor([1, -0.5, 0.0])
    C = t.tensor([1, 0.5, 0.5])
    num_pixels_y = num_pixels_z = 15
    y_limit = z_limit = 0.5

    # Plot triangle & rays
    test_triangle = t.stack([A, B, C], dim=0)
    rays2d = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
    triangle_lines = t.stack([A, B, C, A, B, C], dim=0).reshape(-1, 2, 3)
    render_lines_with_plotly(rays2d, triangle_lines)

    # Calculate and display intersections
    intersects = raytrace_triangle(rays2d, test_triangle)
    print(intersects)
    img = intersects.reshape(num_pixels_y, num_pixels_z).int()
    imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")


# %%
def raytrace_triangle_with_bug(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"]
) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if the triangle intersects that ray.
    '''
    NR = rays.size(0)

    A, B, C = einops.repeat(triangle, "pts dims -> pts NR dims", NR=NR)

    O, D = rays.unbind(1)

    mat = t.stack([- D, B - A, C - A], dim=-1)

    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    vec = O - A

    sol = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(dim=-1)

    return ((u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular)


intersects = raytrace_triangle_with_bug(rays2d, test_triangle)
img = intersects.reshape(num_pixels_y, num_pixels_z).int()
imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")
# %%
if MAIN:
    with open(section_dir / "pikachu.pt", "rb") as f:
        triangles = t.load(f)
# %%

def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"]
) -> Float[Tensor, "nrays"]:
    '''
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    '''
    
    NR = rays.size(0)
    NT = triangles.size(0)

    rays = einops.repeat(rays, 'nrays p d -> nrays nt p d', nt = NT)
    triangles = einops.repeat(triangles, 'ntriangles p d -> nr ntriangles p d', nr = NR)

    assert rays.shape == (NR, NT, 2, 3)
    assert triangles.shape == (NR, NT, 3, 3)

    O, D = rays.unbind(dim = 2)
    assert O.shape == (NR, NT, 3)

    A, B, C = triangles.unbind(dim = 2)
    assert A.shape == (NR, NT, 3)

    mat = torch.stack([-D, B-A, C-A], dim=-1)
    assert mat.shape == (NR, NT, 3, 3)

    vec = O-A

    det = torch.det(mat) # shape (NR, NT)
    singular = abs(det) < 1e-8
    mat[singular] = torch.eye(3)

    sol = torch.linalg.solve(mat, vec) # (NR, NT, 3)
    s,u,v = sol.unbind(dim=-1)
    # Find distance
    dist = torch.sqrt(torch.sum(torch.square(sol), dim=-1))
    # Replace is singular with torch.inf
    valid_mask = ((u >= 0) & (v >= 0) & (u +v <= 1) & (~singular))
    s[~valid_mask] = torch.inf

    #return einops.reduce(s, 'nr nt -> nr', 'min')
    return s.min(dim=-1).values


if MAIN:
    num_pixels_y = 120
    num_pixels_z = 120
    y_limit = z_limit = 1

    rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
    rays[:, 0] = t.tensor([-2, 0.0, 0.0])
    dists = raytrace_mesh(rays, triangles)
    intersects = t.isfinite(dists).view(num_pixels_y, num_pixels_z)
    dists_square = dists.view(num_pixels_y, num_pixels_z)
    img = t.stack([intersects, dists_square], dim=0)

    fig = px.imshow(img, facet_col=0, origin="lower", color_continuous_scale="magma", width=1000)
    fig.update_layout(coloraxis_showscale=False)
    for i, text in enumerate(["Intersects", "Distance"]): 
        fig.layout.annotations[i]['text'] = text
    fig.show()
# %%
a = torch.rand(3)
print(a.min(-1))
print(a.min(-1).values)
# %%
