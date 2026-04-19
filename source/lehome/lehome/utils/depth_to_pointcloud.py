import numpy as np
from scipy.spatial.transform import Rotation as R

# import plotly.graph_objs as go


# ==========================================
# 1. FPS Sampling
# ==========================================
def farthest_point_sampling_with_color(points, colors, n_samples):
    N, D = points.shape
    if N < n_samples:
        indices = np.random.choice(N, n_samples, replace=True)
        return points[indices], colors[indices]

    xyz = points
    centroids = np.zeros((n_samples,), dtype=int)
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)

    for i in range(n_samples):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, axis=0)

    return points[centroids], colors[centroids]


# ==========================================
# 2. Remove Outliers
# ==========================================
def remove_outliers_statistical(points, colors, nb_neighbors=20, std_ratio=2.0):
    from scipy.spatial import cKDTree

    if len(points) == 0:
        return points, colors

    tree = cKDTree(points)
    dists, _ = tree.query(points, k=nb_neighbors)
    mean_dists = np.mean(dists, axis=1)

    global_mean = np.mean(mean_dists)
    global_std = np.std(mean_dists)

    threshold = global_mean + std_ratio * global_std
    mask = mean_dists < threshold

    return points[mask], colors[mask]


# ==========================================
# 3. Pointcloud Generation
# ==========================================
def generate_pointcloud_from_data(
    rgb_image: np.ndarray,
    depth_image: np.ndarray,
    num_points: int = 4096,
    use_fps: bool = True,
    # =========================================================
    # TODO: Calibrate the following parameters for your scene.
    # The fallback values below are legacy single-task examples kept only for
    # cases where task-specific camera metadata is unavailable.
    # They do NOT represent all LeHome tasks.
    # =========================================================
    # Camera intrinsics (fx, fy, cx, cy) for a 640x480 image.
    # Prefer metadata recorded with the dataset over these hardcoded values.
    camera_fx: float = 482.0,
    camera_fy: float = 482.0,
    camera_cx: float = 320.0,
    camera_cy: float = 240.0,
    # Quaternion [w, x, y, z] rotating the camera optical frame
    # into the camera owner's robot-base frame.
    cam_to_base_quat_wxyz: list | None = None,
    # Translation (meters) from camera origin to the camera owner's base origin.
    cam_to_base_translation: np.ndarray | None = None,
    # World position (meters) of the camera owner's robot base.
    world_base_pos: np.ndarray | None = None,
    # Quaternion [w, x, y, z] of the camera owner's robot base in world frame.
    world_base_rot_wxyz: list | None = None,
):
    """
    Convert an RGB-D frame into a colored point cloud in world coordinates.

    Args:
        rgb_image: (H, W, 3) or (H, W, 4) uint8 color image.
        depth_image: (H, W) float32 depth image in **meters**.
        num_points: Number of points to downsample to.
        use_fps: If True, use Farthest Point Sampling; else random sampling.
        camera_fx/fy/cx/cy: Camera intrinsic parameters.
        cam_to_base_quat_wxyz: Rotation from camera to robot-base frame [w,x,y,z].
        cam_to_base_translation: Translation from camera to robot-base frame (m).
        world_base_pos: Position of the corresponding robot base in world frame (m).
        world_base_rot_wxyz: Rotation of the corresponding robot base in world frame [w,x,y,z].

    Returns:
        np.ndarray of shape (N, 6): [x, y, z, r, g, b] float32. RGB values are
        in the range [0, 255].
    """
    # -- Fill in fallback values if task-specific metadata is not provided --
    # TODO: Replace these hardcoded fallback values with task-specific metadata.
    if cam_to_base_quat_wxyz is None:
        cam_to_base_quat_wxyz = [0.1650476, -0.9862856, 0.0, 0.0]  # legacy fallback
    if cam_to_base_translation is None:
        cam_to_base_translation = np.array([0.225, -0.5, 0.6])      # legacy fallback
    if world_base_pos is None:
        world_base_pos = np.array([1.15, -2.3, 0.5], dtype=np.float32)  # legacy fallback
    if world_base_rot_wxyz is None:
        world_base_rot_wxyz = [0.0, 0.0, 0.0, 1.0]                  # identity (no rot)

    fx, fy = camera_fx, camera_fy
    cx, cy = camera_cx, camera_cy

    # Convert quaternion from [w,x,y,z] to [x,y,z,w] for scipy
    quat_xyzw_cam = [
        cam_to_base_quat_wxyz[1], cam_to_base_quat_wxyz[2],
        cam_to_base_quat_wxyz[3], cam_to_base_quat_wxyz[0],
    ]
    translation = np.asarray(cam_to_base_translation)

    quat_xyzw_world = [
        world_base_rot_wxyz[1], world_base_rot_wxyz[2],
        world_base_rot_wxyz[3], world_base_rot_wxyz[0],
    ]
    world_pos = np.asarray(world_base_pos, dtype=np.float32)

    # 1. get valid depth index
    valid_mask = depth_image > 0
    v_idx, u_idx = np.nonzero(valid_mask)
    total_valid = len(v_idx)

    if total_valid == 0:
        return np.zeros((0, 6), dtype=np.float32)

    # 2. pre-sampling to accelerate FPS calculation
    pre_sample_target = max(8192, num_points * 2)
    if total_valid > pre_sample_target:
        sample_indices = np.random.choice(total_valid, pre_sample_target, replace=False)
        v_sample = v_idx[sample_indices]
        u_sample = u_idx[sample_indices]
    else:
        v_sample = v_idx
        u_sample = u_idx

    # 3. Back-project depth to 3D camera space
    Z_sample = depth_image[v_sample, u_sample]
    X_sample = (u_sample - cx) * Z_sample / fx
    Y_sample = (v_sample - cy) * Z_sample / fy
    points_cam = np.stack([X_sample, Y_sample, Z_sample], axis=1)

    # 4. get color (handle RGBA)
    if rgb_image.shape[-1] == 4:
        colors_sample = rgb_image[v_sample, u_sample, :3]
    else:
        colors_sample = rgb_image[v_sample, u_sample]

    # 5. Camera frame -> Arm base frame
    # Isaac Lab/Sim camera is -Y up, apply optical-to-USD rotation first
    r_usd_to_base = R.from_quat(quat_xyzw_cam).as_matrix()
    r_optical_to_usd = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    r_mix = np.dot(r_usd_to_base, r_optical_to_usd)
    points_base = np.dot(points_cam, r_mix.T) + translation

    # 6. Arm base frame -> World frame
    r_world = R.from_quat(quat_xyzw_world).as_matrix()
    points_world = np.dot(points_base, r_world.T) + world_pos

    # 7. Statistical outlier removal
    points_world, colors_sample = remove_outliers_statistical(
        points_world, colors_sample, nb_neighbors=50, std_ratio=1.0
    )

    # 8. Final downsampling (FPS or random)
    if points_world.shape[0] > num_points:
        if use_fps:
            points_world, colors_sample = farthest_point_sampling_with_color(
                points_world, colors_sample, num_points
            )
        else:
            indices = np.random.choice(points_world.shape[0], num_points, replace=False)
            points_world = points_world[indices]
            colors_sample = colors_sample[indices]

    # Combine to (N, 6): [x, y, z, r, g, b], float32
    points_with_color = np.concatenate(
        [points_world.astype(np.float32), colors_sample.astype(np.float32)], axis=1
    )
    return points_with_color
