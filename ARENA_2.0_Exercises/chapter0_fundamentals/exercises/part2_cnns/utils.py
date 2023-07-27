import sys, os
import plotly.express as px
import einops
import numpy as np
from pathlib import Path

chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part2_cnns"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

def display_array_as_img(img_array):
    '''
    Displays a numpy array as an image
    
    Two options:
        img_array.shape = (height, width) -> interpreted as monochrome
        img_array.shape = (3, height, width) -> interpreted as RGB
    '''
    shape = img_array.shape
    assert len(shape) == 2 or (shape[0] == 3 and len(shape) == 3), "Incorrect format (see docstring)"
    
    if len(shape) == 3:
        img_array = einops.rearrange(img_array, "c h w -> h w c")
    height, width = img_array.shape[:2]
    
    fig = px.imshow(img_array, zmin=0, zmax=255, color_continuous_scale='gray')
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(coloraxis_showscale=False, margin=dict.fromkeys("tblr", 0), height=height, width=width)
    fig.show(config=dict(displayModeBar=False))

arr = np.load(section_dir / "numbers.npy")

arr1_soln = einops.rearrange(arr, "b c h w -> c h (b w)")
arr2_soln = einops.repeat(arr[0], "c h w -> c (2 h) w")
arr3_soln = einops.repeat(arr[0:2], "b c h w -> c (b h) (2 w)")
arr4_soln = einops.repeat(arr[0], "c h w -> c (h 2) w")
arr5_soln = einops.rearrange(arr[0], "c h w -> h (c w)")
arr6_soln = einops.rearrange(arr, "(b1 b2) c h w -> c (b1 h) (b2 w)", b1=2)
arr7_soln = einops.reduce(arr.astype(float), "b c h w -> h (b w)", "max").astype(int)
arr8_soln = einops.reduce(arr.astype(float), "b c h w -> h w", "min").astype(int)
arr9_soln = einops.rearrange(arr[1], "c h w -> c w h")
arr10_soln = einops.reduce(arr, "(b1 b2) c (h h2) (w w2) -> c (b1 h) (b2 w)", "max", h2=2, w2=2, b1=2)

def display_soln_array_as_img(i):
    assert isinstance(i, int)
    assert 1 <= i <= 10
    return display_array_as_img([arr1_soln, arr2_soln, arr3_soln, arr4_soln, arr5_soln, arr6_soln, arr7_soln, arr8_soln, arr9_soln, arr10_soln][i-1])