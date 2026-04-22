"""Dataset replay utility functions for replaying recorded episodes."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
from typing import Dict, Optional, Tuple, Any
import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lehome.utils.logger import get_logger

from lehome.utils.record import RateLimiter

from .common import stabilize_robot

logger = get_logger(__name__)


def build_episode_row_index_map(
    hf_dataset: Any,
    start_episode: int,
    end_episode: int,
) -> Dict[int, list[int]]:
    """Build episode-to-row-index mapping in a single pass.

    Args:
        hf_dataset: HuggingFace dataset object.
        start_episode: Inclusive start episode index.
        end_episode: Exclusive end episode index.

    Returns:
        Dictionary mapping episode index to list of dataset row indices.
    """
    episode_to_rows: Dict[int, list[int]] = defaultdict(list)
    episode_column = hf_dataset["episode_index"]

    for row_idx, episode_value in enumerate(episode_column):
        episode_idx = int(episode_value.item()) if hasattr(episode_value, "item") else int(episode_value)
        if start_episode <= episode_idx < end_episode:
            episode_to_rows[episode_idx].append(row_idx)

    return dict(episode_to_rows)


def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments for dataset replay.

    Ensures the dataset exists, contains a valid info.json, and that the
    requested episode range is within bounds.

    Args:
        args: Command-line arguments containing replay configuration.

    Raises:
        ValueError: If dataset path is invalid, dataset is empty, or episode
            range is invalid.
    """
    dataset_path = Path(args.dataset_root)
    if not dataset_path.exists():
        raise ValueError(f"Dataset root does not exist: {args.dataset_root}")

    info_json = dataset_path / "meta" / "info.json"
    if not info_json.exists():
        raise ValueError(f"Dataset info.json not found: {info_json}")

    try:
        with open(info_json, "r") as f:
            info = json.load(f)
        total_episodes = info.get("total_episodes", 0)
        if total_episodes == 0:
            raise ValueError(
                f"Dataset is empty (total_episodes=0). "
                f"Please use a dataset with recorded episodes: {args.dataset_root}"
            )
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse info.json: {e}")

    if args.num_replays < 1:
        raise ValueError(f"num_replays must be >= 1, got {args.num_replays}")

    # Validate episode range
    if args.start_episode < 0:
        raise ValueError(f"start_episode must be >= 0, got {args.start_episode}")

    if args.end_episode is not None:
        if args.end_episode < 0:
            raise ValueError(f"end_episode must be >= 0, got {args.end_episode}")
        if args.end_episode <= args.start_episode:
            raise ValueError(
                f"end_episode ({args.end_episode}) must be > start_episode ({args.start_episode})"
            )


def load_dataset(dataset_root: str) -> LeRobotDataset:
    """Load the LeRobotDataset from the specified root directory.

    Args:
        dataset_root: Root directory path of the dataset to load.

    Returns:
        Loaded LeRobotDataset instance.

    Raises:
        FileNotFoundError: If dataset data directory is not found.
        ValueError: If dataset has no episodes.
    """
    logger.info(f"Loading dataset from: {dataset_root}")

    dataset_path = Path(dataset_root)
    data_dir = dataset_path / "data"
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset data directory not found: {data_dir}. "
            f"The dataset might be incomplete."
        )

    try:
        dataset = LeRobotDataset(repo_id="replay_source", root=dataset_root)
        logger.info(
            f"Dataset loaded: {dataset.num_episodes} episodes, {dataset.num_frames} frames"
        )

        if dataset.num_episodes == 0:
            raise ValueError(f"Dataset has no episodes (num_episodes=0).")

        return dataset
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise



def load_initial_pose(
    dataset_root: str, episode_index: int
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Load the initial object pose for a specific episode from metadata.

    The pose is essential for replaying the exact same initial conditions
    encountered during recording.

    Args:
        dataset_root: Root directory of the dataset.
        episode_index: The specific episode index to load.

    Returns:
        A tuple of (object_initial_pose, variant_name).
        Returns (None, None) if no metadata is found.
    """
    metadata_file = Path(dataset_root) / "meta" / "episode_metadata.json"
    if not metadata_file.exists():
        metadata_file = Path(dataset_root) / "meta" / "garment_info.json"

    if not metadata_file.exists():
        return None, None

    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        episode_key = str(episode_index)
        # Iterate through variants to find the episode entry
        for variant_name, episodes in data.items():
            if episode_key in episodes:
                pose_data = episodes[episode_key].get("object_initial_pose")
                if pose_data is not None:
                    if isinstance(pose_data, dict) and "Garment" in pose_data:
                        return pose_data, variant_name
                    return {"Garment": pose_data}, variant_name

        return None, None
    except Exception as e:
        logger.warning(f"Failed to load initial pose for episode {episode_index}: {e}")
        return None, None


def create_replay_dataset(
    args: argparse.Namespace, source_dataset: LeRobotDataset
) -> Tuple[Optional[LeRobotDataset], Optional[Path]]:
    """Create a new dataset to save the replayed (and potentially modified) frames.

    Args:
        args: Command-line arguments containing output paths.
        source_dataset: The original dataset providing the features schema.

    Returns:
        A tuple of (replay_dataset, json_path).
    """
    if args.output_root is None:
        return None, None

    output_path = Path(args.output_root)
    source_folder_name = Path(args.dataset_root).name
    root = output_path / source_folder_name
    
    # Clean output directory for a fresh replay
    if root.exists():
        logger.info(f"Target path {root} already exists. Cleaning up...")
        shutil.rmtree(root)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    # Inherit features from source
    features = source_dataset.meta.features
    if args.disable_depth and "observation.top_depth" in features:
        features = {k: v for k, v in features.items() if k != "observation.top_depth"}

    logger.info(f"Creating replay output dataset at: {root}")
    replay_dataset = LeRobotDataset.create(
        repo_id="replay_output",
        fps=source_dataset.fps,
        root=root,
        use_videos=True,
        image_writer_threads=8,
        image_writer_processes=0,
        features=features,
    )

    metadata_path = replay_dataset.root / "meta" / "episode_metadata.json"
    return replay_dataset, metadata_path


def compute_action_from_ee_pose(
    env: DirectRLEnv,
    frame_data: Dict[str, torch.Tensor],
    ik_solver: Any,
    is_bimanual: bool,
    args: argparse.Namespace,
    ik_stats: Dict[str, Any],
    device: str = "cpu",
) -> Optional[torch.Tensor]:
    """Compute target joint angles from Cartesian End-Effector (EE) pose using Inverse Kinematics.

    This function facilitates 'Cartesian Replay', allowing recorded datasets to be played back
    by converting recorded EE poses back into joint actions. This is useful for adapting
    trajectories to slightly different robot configurations or verifying IK consistency.

    Args:
        env: The Isaac Lab environment instance.
        frame_data: A single frame's data from the LeRobotDataset containing 'action.ee_pose'.
        ik_solver: The RobotKinematics solver loaded with the robot's URDF.
        is_bimanual: True if the task involves dual arms (affects indexing).
        args: Command-line arguments (used for 'ee_state_unit').
        ik_stats: A dictionary to track IK success, total attempts, and fallbacks.
        device: The device (cpu/cuda) where the resulting action tensor should reside.

    Returns:
        A torch.Tensor of shape (1, num_actions) containing the solved joint angles, 
        or None if the IK solver fails or required data is missing.
    """
    from lehome.utils import compute_joints_from_ee_pose

    try:
        if "action.ee_pose" not in frame_data:
            ik_stats["total"] += 1
            ik_stats["fallback"] += 1
            return None

        action_ee_pose = frame_data["action.ee_pose"].cpu().numpy()
        current_state = frame_data["observation.state"].cpu().numpy().flatten()

        if is_bimanual:
            # Dual-arm IK
            res_left = compute_joints_from_ee_pose(ik_solver, current_state[:6], action_ee_pose[:8], args.ee_state_unit)
            res_right = compute_joints_from_ee_pose(ik_solver, current_state[6:12], action_ee_pose[8:16], args.ee_state_unit)
            if res_left is None or res_right is None:
                ik_stats["fallback"] += 1
                return None
            action_joints = np.concatenate([res_left, res_right], axis=0)
        else:
            # Single-arm IK
            action_joints = compute_joints_from_ee_pose(ik_solver, current_state, action_ee_pose, args.ee_state_unit)
            if action_joints is None:
                ik_stats["fallback"] += 1
                return None

        ik_stats.update({"total": ik_stats["total"]+1, "success": ik_stats["success"]+1})
        return torch.from_numpy(action_joints).float().to(device).unsqueeze(0)
    except Exception as e:
        logger.warning(f"IK failed: {e}")
        ik_stats["total"] += 1
        ik_stats["fallback"] += 1
        return None


def replay_episode(
    env: DirectRLEnv,
    episode_data: Any,
    rate_limiter: Optional[RateLimiter],
    initial_pose: Optional[Dict[str, Any]],
    args: argparse.Namespace,
    replay_dataset: Optional[LeRobotDataset] = None,
    disable_depth: bool = False,
    ik_solver: Optional[Any] = None,
    is_bimanual: bool = False,
    ik_stats: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    task_description: str = "",
) -> bool:
    """Replay a single episode step-by-step from recorded data.

    Optionally computes actions via Inverse Kinematics if `--use_ee_pose` is enabled.
    Can also save the replayed frames to a new dataset.

    Args:
        env: The environment instance.
        episode_data: Frame-by-frame recorded data for the episode.
        rate_limiter: Limiter to control replay speed.
        initial_pose: Initial object transformation to apply before replay.
        args: Command-line arguments.
        replay_dataset: Optional dataset instance to save replayed frames.
        disable_depth: Whether to skip depth observation saving.
        ik_solver: RobotKinematics solver for IK computations.
        is_bimanual: Whether the episode uses two arms.
        ik_stats: Dictionary to track IK success/fallback statistics.
        device: Torch device (cpu/cuda).
        task_description: Description string to save with each frame.

    Returns:
        True if the episode achieved the success condition, False otherwise.
    """
    try:
        # 1. Reset and set initial conditions
        env.reset()
        if initial_pose is not None:
            if hasattr(env, "set_all_pose"):
                try:
                    env.set_all_pose(initial_pose)
                except Exception as e:
                    logger.error(f"Failed to set initial pose: {e}")
            else:
                logger.warning("Env does not support set_all_pose, skip setting initial pose.")

        # 2. Stabilize environment before starting replay
        stabilize_robot(env)
        success_achieved = False

        # 3. Step-by-step replay loop
        for idx in range(len(episode_data)):
            if rate_limiter:
                rate_limiter.sleep(env)

            # Determine action: IK-based or direct joint-based
            if args.use_ee_pose and ik_solver is not None:
                action = compute_action_from_ee_pose(env, episode_data[idx], ik_solver, is_bimanual, args, ik_stats, device)
                if action is None:
                    # Fallback to recorded joint action if IK fails
                    action = episode_data[idx]["action"].to(device).unsqueeze(0)
            else:
                action = episode_data[idx]["action"].to(device).unsqueeze(0)

            env.step(action)

            # Optional: Save replayed frame
            if replay_dataset is not None:
                obs = env._get_observations()
                if disable_depth and "observation.top_depth" in obs:
                    obs = {k: v for k, v in obs.items() if k != "observation.top_depth"}
                replay_dataset.add_frame({**obs, "task": task_description})

            # Track success reaching
            try:
                if env._get_success().item():
                    success_achieved = True
            except: pass

        return success_achieved
    except Exception as e:
        logger.error(f"Error replaying episode: {e}")
        return False


def append_episode_initial_pose(
    json_path: Path,
    episode_idx: int,
    object_initial_pose: Dict[str, Any],
    variant_name: Optional[str] = None,
) -> None:
    """Append initial pose and metadata for a single episode to a hierarchical JSON file.

    This function acts as a wrapper for the core utility in `lehome.utils.record`.
    It ensures that when replaying and saving a dataset, the environment's initial
    conditions (object position and variant type) are preserved or updated
    in the new dataset's 'episode_metadata.json'.

    Args:
        json_path: Path to the 'episode_metadata.json' file in the target dataset.
        episode_idx: The integer index of the current episode.
        object_initial_pose: The initial state dictionary derived from the environment.
        variant_name: The name of the object variant (e.g., 'shirt_01').
    """
    from lehome.utils.record import append_episode_initial_pose as append_pose
    append_pose(json_path, episode_idx, object_initial_pose, variant_name=variant_name)


def replay(args: argparse.Namespace) -> None:
    """Main entry point for replaying a dataset.

    Coordinates dataset loading, environment setup (including auto-detecting
    variants), and iterating through episodes for replay and evaluation.

    Args:
        args: Parsed command-line arguments.
    """
    validate_args(args)
    dataset = load_dataset(args.dataset_root)

    device = getattr(args, "device", "cpu")
    task_desc = getattr(args, "task_description", "")

    # End-Effector Controller Setup (optional)
    ik_solver = None
    is_bimanual = False
    ik_stats = {"total": 0, "success": 0, "fallback": 0, "errors": []}

    if args.use_ee_pose:
        if "observation.ee_pose" not in dataset.meta.features:
            raise ValueError("Dataset lacks recorded end-effector poses (observation.ee_pose).")
        
        from lehome.utils import RobotKinematics
        state_dim = dataset.meta.features["observation.state"]["shape"][0]
        is_bimanual = (state_dim == 12)
        joint_names = dataset.meta.features["observation.state"]["names"]
        # Standardize joint names for the IK solver (removing side prefixes)
        solver_names = [n.replace("left_", "").replace("right_", "") for n in joint_names[:5]]

        ik_solver = RobotKinematics(
            str(args.ee_urdf_path), 
            target_frame_name="gripper_frame_link", 
            joint_names=solver_names
        )
        logger.info(f"IK solver loaded ({'dual' if is_bimanual else 'single'}-arm)")

    # 1. Environment Configuration
    logger.info(f"Creating environment: {args.task}")
    env_cfg = parse_env_cfg(args.task, device=device)

    if hasattr(env_cfg, "garment_name") and env_cfg.garment_name is None:
        garment_info_file = Path(args.dataset_root) / "meta" / "garment_info.json"
        if garment_info_file.exists():
            with open(garment_info_file, "r", encoding="utf-8") as f:
                garment_info = json.load(f)
            garment_name = next(iter(garment_info))
            env_cfg.garment_name = garment_name
            logger.info(f"Loaded garment name from dataset metadata: {garment_name}")

    if hasattr(env_cfg, "use_random_seed"):
        env_cfg.use_random_seed = args.use_random_seed

    if args.use_random_seed:
        if hasattr(env_cfg, "seed"):
            env_cfg.seed = None
        if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "seed"):
            env_cfg.sim.seed = None
        logger.info("Using random seed (no fixed seed)")
    else:
        if hasattr(env_cfg, "seed"):
            env_cfg.seed = args.seed
        if hasattr(env_cfg, "random_seed"):
            env_cfg.random_seed = args.seed
        if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "seed"):
            env_cfg.sim.seed = args.seed
        logger.info(f"Using fixed random seed: {args.seed}")

    # 2. Environment Creation
    env: DirectRLEnv = gym.make(args.task, cfg=env_cfg).unwrapped


    if hasattr(env, "initialize_obs"):
        env.initialize_obs()

    logger.info(
        f"[Render Debug] has_gui={env.sim.has_gui()}, "
        f"has_rtx_sensors={env.sim.has_rtx_sensors()}, "
        f"render_mode={env.sim.render_mode}"
    )

    # 3. Setup Replay Infrastructure
    rate_limiter = RateLimiter(args.step_hz) if args.step_hz > 0 else None
    replay_dataset, metadata_path = create_replay_dataset(args, dataset)

    # Determine episode range
    start_idx = args.start_episode
    end_idx = min(
        args.end_episode if args.end_episode is not None else dataset.num_episodes, 
        dataset.num_episodes
    )
    total_episodes = end_idx - start_idx

    total_attempts = 0
    total_successes = 0
    saved_episodes = 0

    logger.info("Building episode row index map for replay...")
    episode_row_index_map = build_episode_row_index_map(
        dataset.hf_dataset,
        start_idx,
        end_idx,
    )

    # 4. Main Replay Loop
    try:
        for episode_idx in range(start_idx, end_idx):
            display_num = episode_idx - start_idx + 1
            logger.info(f"--- Episode {display_num}/{total_episodes} (index {episode_idx}) ---")

            # Load original conditions
            initial_pose, variant_name = load_initial_pose(args.dataset_root, episode_idx)
            episode_row_indices = episode_row_index_map.get(episode_idx, [])
            if not episode_row_indices:
                logger.warning(f"Episode {episode_idx} has no frames. Skipping.")
                continue

            try:
                episode_data = dataset.hf_dataset.select(episode_row_indices)
            except Exception as e:
                logger.error(f"Failed to load frames for episode {episode_idx}: {e}")
                continue

            # Performance repeated replays per episode if requested
            for r_idx in range(args.num_replays):
                total_attempts += 1
                if replay_dataset:
                    replay_dataset.clear_episode_buffer()

                success = replay_episode(
                    env=env, episode_data=episode_data, rate_limiter=rate_limiter,
                    initial_pose=initial_pose, args=args, replay_dataset=replay_dataset,
                    disable_depth=args.disable_depth, ik_solver=ik_solver, is_bimanual=is_bimanual,
                    ik_stats=ik_stats, device=device, task_description=task_desc
                )

                if success:
                    total_successes += 1
                logger.info(f"  Replay {r_idx+1}/{args.num_replays}: {'[SUCCESS]' if success else '[FAILED]'}")

                # Save replayed data if requested
                if replay_dataset and (not args.save_successful_only or success):
                    replay_dataset.save_episode()
                    append_episode_initial_pose(
                        metadata_path, saved_episodes, initial_pose, variant_name=variant_name
                    )
                    saved_episodes += 1
                    logger.info(f"  Saved as episode {saved_episodes-1} in output dataset.")
                elif replay_dataset:
                    replay_dataset.clear_episode_buffer()

    finally:
        # 5. Cleanup and Summary
        if replay_dataset:
            replay_dataset.clear_episode_buffer()
            replay_dataset.finalize()

        if total_attempts > 0:
            success_rate = 100.0 * total_successes / total_attempts
            logger.info("=" * 60)
            logger.info("🎬 Replay Summary")
            logger.info("=" * 60)
            logger.info(f"  Total Attempts: {total_attempts}")
            logger.info(f"  Total Successes: {total_successes}")
            logger.info(f"  Success Rate: {success_rate:.1f}%")
            if args.use_ee_pose:
                logger.info(f"  IK Stats: {ik_stats['success']}/{ik_stats['total']} successes")
            logger.info("=" * 60)

        env.close()
