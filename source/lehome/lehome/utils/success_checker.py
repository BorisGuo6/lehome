import math
import numpy as np
import torch
from lehome.utils.logger import get_logger

logger = get_logger(__name__)


def step_interval(interval=50):
    """Factory function: creates a customizable step interval decorator"""

    def decorator(func):
        call_count = 0

        def wrapper(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count % interval == 0:
                return func(*args, **kwargs)
            else:
                return False

        return wrapper

    return decorator


def calculate_distance(point_a, point_b):
    # calculate distance
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    return np.linalg.norm(point_a - point_b)


def get_object_particle_position(particle_object, index_list):
    try:
        _, mesh_points, _, _ = particle_object.get_current_mesh_points()
    except Exception as e1:
        try:
            logger.error(f"Error in get_object_particle_position: {e1}")
            mesh_points = (
                particle_object._cloth_prim_view.get_world_positions()
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )
        except Exception as e2:
            logger.error(f"Error in get_object_particle_position: {e2}")
            return None

    mesh_points = np.asarray(mesh_points)
    if mesh_points.ndim != 2 or mesh_points.shape[0] == 0:
        return None

    if any(index >= mesh_points.shape[0] or index < 0 for index in index_list):
        return None

    position = mesh_points * 100
    return [tuple(position[index]) for index in index_list]


@step_interval(interval=50)
def success_checker_fold(
    particle_object, index_list=[8077, 1711, 2578, 3942, 8738, 588]
):
    p = get_object_particle_position(particle_object, index_list)
    if p is None:
        return False
    success = (
        calculate_distance(p[0], p[4]) <= 10
        and calculate_distance(p[2], p[3]) <= 16
        and calculate_distance(p[1], p[5]) <= 10
    )
    return bool(success)


@step_interval(interval=50)
def success_checker_fling(
    particle_object, index_list=[8077, 1711, 2578, 3942, 8738, 588]
):
    p = get_object_particle_position(particle_object, index_list)
    if p is None:
        return False

    def xy_distance(a, b):
        return np.linalg.norm(np.array(a[:2]) - np.array(b[:2]))

    def z_distance(a, b):
        return abs(a[2] - b[2])

    success = (
        xy_distance(p[0], p[4]) > 18
        and z_distance(p[0], p[4]) < 2
        and xy_distance(p[1], p[5]) > 18
        and z_distance(p[1], p[5]) < 2
    )

    return bool(success)


@step_interval(interval=30)
def success_checker_burger(beef_pos, plate_pos):
    diff_xy = beef_pos[:, :2] - plate_pos[:, :2]
    dist_xy = torch.linalg.norm(diff_xy, dim=-1)

    # z distance
    diff_z = torch.abs(beef_pos[:, 2] - plate_pos[:, 2])

    # success condition: both xy < 0.05 and z < 0.1
    success_mask = (dist_xy < 0.045) & (diff_z < 0.03)
    success = success_mask.any().item()

    return bool(success)


@step_interval(interval=6)
def success_checker_cut(sausage_count: int) -> bool:
    return sausage_count >= 2
