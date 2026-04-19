"""Evaluation utility functions for policy evaluation in LeHome environments."""

import os
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import torch
from torch import Tensor
from lehome.utils.logger import get_logger

logger = get_logger(__name__)


def convert_ee_pose_to_joints(
    ee_pose_action: torch.Tensor,
    current_joints: torch.Tensor,
    solver: Any,
    is_bimanual: bool,
    state_unit: str = "rad",
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Convert end-effector pose action to joint angles using IK.

    Args:
        ee_pose_action: EE pose from policy. Single-arm: (8,), Bimanual: (16,).
        current_joints: Current joint positions for IK warm start.
            Single-arm: (6,), Bimanual: (12,).
        solver: RobotKinematics instance.
        is_bimanual: Whether dual-arm or single-arm.
        state_unit: Joint angle unit, either 'rad' or 'deg'.
        device: Target device for output tensor.

    Returns:
        Joint angles tensor (same shape as current_joints).
    """
    from lehome.utils import compute_joints_from_ee_pose

    ee_pose_np = ee_pose_action.cpu().numpy()
    current_joints_np = current_joints.cpu().numpy()

    if is_bimanual:
        left_joints = compute_joints_from_ee_pose(
            solver,
            current_joints_np[:6],
            ee_pose_np[:8],
            state_unit,
            orientation_weight=0.01,
        )
        right_joints = compute_joints_from_ee_pose(
            solver,
            current_joints_np[6:12],
            ee_pose_np[8:16],
            state_unit,
            orientation_weight=0.01,
        )

        if left_joints is None:
            logger.warning("Left arm IK failed, using current joints")
            left_joints = current_joints_np[:6]
        if right_joints is None:
            logger.warning("Right arm IK failed, using current joints")
            right_joints = current_joints_np[6:12]

        joint_angles = np.concatenate([left_joints, right_joints])
    else:
        joint_angles = compute_joints_from_ee_pose(
            solver, current_joints_np, ee_pose_np, state_unit, orientation_weight=0.01
        )

        if joint_angles is None:
            logger.warning("IK failed, using current joints")
            joint_angles = current_joints_np

    return torch.from_numpy(joint_angles).float().to(device)


def preprocess_observation(
    obs_dict: Dict[str, Union[np.ndarray, Dict[str, Any]]],
    device: torch.device,
    task_description: str,
) -> Dict[str, Tensor]:
    """Preprocess observation dictionary into batched PyTorch tensors.

    Converts numpy arrays from the environment into batched tensors on the
    target device. Images are transposed from (H, W, C) to (C, H, W) and
    normalized to [0, 1].

    Args:
        obs_dict: Observation dictionary from environment containing numpy arrays.
        device: Target PyTorch device.
        task_description: Task description string added to the batch as 'task'.

    Returns:
        Dictionary with same structure, values as batched PyTorch tensors on device.

    Raises:
        TypeError: If a value in obs_dict is not a numpy array or dict.
    """
    processed_dict = {}
    for key, value in obs_dict.items():
        if isinstance(value, dict):
            # Recursively handle nested dictionaries
            processed_dict[key] = preprocess_observation(value, device, task_description)
            continue

        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"Expected numpy array for key '{key}', but got {type(value)}"
            )
        processed_value = value

        # Process image data: (H, W, C) -> (C, H, W), normalize to [0, 1]
        if processed_value.ndim == 3 and processed_value.shape[-1] == 3:
            assert (
                processed_value.dtype == np.uint8
            ), f"Image for key '{key}' expected np.uint8, got {processed_value.dtype}"
            processed_value = processed_value.astype(np.float32) / 255.0
            processed_value = np.transpose(processed_value, (2, 0, 1))

        batched_value = np.expand_dims(processed_value, axis=0)
        processed_dict[key] = torch.as_tensor(
            batched_value, dtype=torch.float32, device=device
        )
        processed_dict["task"] = task_description
    return processed_dict


def save_videos_from_observations(
    all_episode_frames: Dict[str, List[np.ndarray]],
    save_dir: str,
    episode_idx: int,
    success: torch.Tensor,
    variant_name: str = "",
    fps: int = 30,
) -> None:
    """Save captured observation frames as MP4 videos.

    Videos are organized into 'success' and 'failure' subdirectories.

    Args:
        all_episode_frames: Dictionary mapping camera key to list of RGB frames.
        save_dir: Root directory to save videos.
        episode_idx: Episode index used in output filename.
        success: Boolean tensor indicating whether the episode succeeded.
        variant_name: Optional variant/object name prefix for the output filename.
        fps: Frames per second for the output video.
    """
    if success.item():
        target_dir = os.path.join(save_dir, "success")
    else:
        target_dir = os.path.join(save_dir, "failure")

    os.makedirs(target_dir, exist_ok=True)

    for key, frames in all_episode_frames.items():
        if len(frames) == 0:
            continue
        h, w, c = frames[0].shape
        prefix = f"{variant_name}_" if variant_name else ""
        out_path = os.path.join(
            target_dir, f"{prefix}episode{episode_idx}_{key.replace('.', '_')}.mp4"
        )
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
        writer.release()
        logger.info(f"Saved video: {out_path}")


def calculate_and_print_metrics(metrics: List[Dict[str, Any]]) -> None:
    """Calculate and print aggregated performance metrics.

    Args:
        metrics: List of episode metric dictionaries, each containing
            'return', 'length', and 'success' keys.
    """
    if not metrics:
        logger.info("[Results] No evaluation metrics were collected.")
        return

    total_returns = [m["return"] for m in metrics]
    total_successes = [1 if m["success"] else 0 for m in metrics]

    avg_return = np.mean(total_returns)
    # Use sample standard deviation (ddof=1) for better estimation with small samples
    std_return = np.std(total_returns, ddof=1) if len(total_returns) > 1 else 0.0
    success_rate = np.mean(total_successes)

    logger.info("=" * 50)
    logger.info("Evaluation Results Summary")
    logger.info("=" * 50)
    logger.info(f"Total Episodes: {len(metrics)}")
    logger.info(f"Average Return: {avg_return:.2f} ± {std_return:.2f}")
    logger.info(f"Success Rate: {success_rate:.2%}")
    logger.info("=" * 50)
