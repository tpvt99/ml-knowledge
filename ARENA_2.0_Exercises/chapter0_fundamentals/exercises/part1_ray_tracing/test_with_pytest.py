import os, sys
CHAPTER = r"chapter0_fundamentals"
chapter_dir = r"./" if CHAPTER in os.listdir() else os.getcwd().split(CHAPTER)[0]
sys.path.append(chapter_dir + f"{CHAPTER}/exercises")

import torch as t
import pytest
import part1_ray_tracing.solutions as solutions
import part1_ray_tracing.answers as answers # type: ignore

# Get a basic set of rays and segments
ray_segment_batch = (
    solutions.rays1d,
    solutions.segments
)

# Get a special case of (ray, segment), where they are parallel
ray_segment_special_case = (
    t.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 1.0]]]),
    t.tensor([[[0.0, 2.0, 2.0], [0.0, 4.0, 4.0]]])
)

# Get a special case of (rays, segments) set, where they are parallel and 
ray_segment_batch_special_case = (
    t.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 1.0]], [[0.0, 0.0, 0.0], [1.0, -10.0, 0.0]]]),
    t.tensor([[[0.0, 2.0, 2.0], [0.0, 4.0, 4.0]], [[1.0, -12.0, 0.0], [1.0, -6.0, 0.0]]])
)

# Get a single triangle and two rays (one intersects, one doesn't)
triangle_and_rays = (
    t.tensor([2, 0.0, -1.0]),
    t.tensor([2, -1.0, 0.0]),
    t.tensor([2, 1.0, 1.0]),
    t.tensor([[[0.0, 0.0, 0.0], [1.0000, 0.3333, 0.3333]], [[0.0, 0.0, 0.0], [1.0, 1.0, -1.0]]])
)


@pytest.mark.parametrize("rays, segments", [
    ray_segment_batch, ray_segment_special_case
])
def test_intersect_ray_1d(rays, segments):
    '''Tests intersect_ray_1d, by looping over all (rays, segments) and finding the intersecting pairs.
    '''
    for segment in segments:
        for ray in rays:
            assert solutions.intersect_ray_1d(ray, segment) == answers.intersect_ray_1d(ray, segment)

@pytest.mark.parametrize("rays, segments", [
    ray_segment_batch, ray_segment_batch_special_case
])
def test_intersect_rays_1d(rays, segments):
    '''Tests intersect_rays_1d (which performs batched computation).
    '''
    actual = answers.intersect_rays_1d(rays, segments)
    expected = solutions.intersect_rays_1d(rays, segments)
    t.testing.assert_close(actual, expected)

@pytest.mark.parametrize("A, B, C, rays", [
    triangle_and_rays
])
def test_triangle_ray_intersects(A, B, C, rays):
    '''Tests triangle_ray_intersects, by iterating through the rays and finding which intersect with the triangle.
    '''
    for (O, D) in rays:
        assert solutions.triangle_ray_intersects(A, B, C, O, D) == answers.triangle_ray_intersects(A, B, C, O, D)
