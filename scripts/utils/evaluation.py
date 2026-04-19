import os
import argparse
import gymnasium as gym
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from isaaclab.envs import DirectRLEnv
from isaaclab_tasks.utils import parse_env_cfg

from scripts.eval_policy import PolicyRegistry
from scripts.eval_policy.base_policy import BasePolicy

from scripts.utils.eval_utils import (
    convert_ee_pose_to_joints,
    save_videos_from_observations,
    calculate_and_print_metrics,
)

from lehome.utils.record import (
    RateLimiter,
    get_next_experiment_path_with_gap,
    append_episode_initial_pose,
)
from lehome.utils.logger import get_logger
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from .common import stabilize_robot

logger = get_logger(__name__)


def run_evaluation_loop(
    env: DirectRLEnv,
    policy: BasePolicy,
    args: argparse.Namespace,
    ee_solver: Optional[Any] = None,
    is_bimanual: bool = False,
) -> List[Dict[str, Any]]:
    """
    Core evaluation loop.
    Refactored to be agnostic of specific task types or implementations.
    """

    # --- Dataset Recording Setup (Optional) ---
    eval_dataset = None
    json_path = None
    episode_index = 0
    if args.save_datasets:
        features = None
        if args.dataset_root and Path(args.dataset_root).exists():
            source_dataset = LeRobotDataset(repo_id="collected_dataset", root=Path(args.dataset_root))
            features = dict(source_dataset.meta.features)
            fps = source_dataset.fps
        else:
            fps = 30  # Default FPS if no source dataset is provided
            action_names = [
                "shoulder_pan", "shoulder_lift", "elbow_flex",
                "wrist_flex", "wrist_roll", "gripper",
            ]
            if is_bimanual:
                left_names = [f"left_{n}" for n in action_names]
                right_names = [f"right_{n}" for n in action_names]
                joint_names = left_names + right_names
            else:
                joint_names = action_names
            dim = len(joint_names)
            features = {
                "observation.state": {
                    "dtype": "float32",
                    "shape": (dim,),
                    "names": joint_names,
                },
                "action": {
                    "dtype": "float32",
                    "shape": (dim,),
                    "names": joint_names,
                },
            }
            image_keys = ["top_rgb", "left_rgb", "right_rgb"] if is_bimanual else ["top_rgb", "wrist_rgb"]
            for key in image_keys:
                features[f"observation.images.{key}"] = {
                    "dtype": "video",
                    "shape": (480, 640, 3),
                    "names": ["height", "width", "channels"],
                }
        root_path = Path(args.eval_dataset_path)
        eval_dataset = LeRobotDataset.create(
            repo_id="lehome_eval",
            fps=fps,
            root=get_next_experiment_path_with_gap(root_path),
            use_videos=True,
            image_writer_threads=8,
            image_writer_processes=0,
            features=features,
        )
        # Use a generic name for metadata
        json_path = eval_dataset.root / "meta" / "episode_metadata.json"

    all_episode_metrics = []
    logger.info(f"Starting evaluation: {args.num_episodes} episodes")
    rate_limiter = RateLimiter(args.step_hz) if args.step_hz > 0 else None
    total_success_count = 0

    for i in range(args.num_episodes):
        # 1. Reset Environment & Policy
        env.reset()
        policy.reset()

        # Stabilize environment (physics settling)
        stabilize_robot(env)

        # 2. Initial Observation
        object_initial_pose = None
        if args.save_datasets and hasattr(env, "get_all_pose"):
            object_initial_pose = env.get_all_pose()

        observation_dict = env._get_observations()

        # Prepare for video recording
        episode_frames = (
            {k: [] for k in observation_dict.keys() if "images" in k}
            if args.save_video
            else {}
        )

        episode_return = 0.0
        episode_length = 0
        extra_steps = 0
        success_flag = False
        success = torch.tensor(False)

        for st in range(args.max_steps):
            if rate_limiter:
                rate_limiter.sleep(env)

            # 3. Policy Inference (The core abstraction)
            # Input: Numpy Dict -> Output: Numpy Array
            action_np = policy.select_action(observation_dict)

            # 4. Prepare Action for Environment (Tensor)
            # Convert numpy action to tensor for Isaac Lab
            action = torch.from_numpy(action_np).float().to(args.device).unsqueeze(0)

            # 5. Inverse Kinematics (Optional Helper Logic)
            # If policy outputs EE pose but env needs joints
            if args.use_ee_pose and ee_solver is not None:
                current_joints = (
                    torch.from_numpy(observation_dict["observation.state"])
                    .float()
                    .to(args.device)
                )
                action = convert_ee_pose_to_joints(
                    ee_pose_action=action.squeeze(0),
                    current_joints=current_joints,
                    solver=ee_solver,
                    is_bimanual=is_bimanual,
                    state_unit="rad",
                    device=args.device,
                ).unsqueeze(0)

            # 6. Step Environment
            env.step(action)

            # Check success
            if not success_flag:
                # Attempt to query success status from the environment if the method is available.
                # We catch AttributeError/NotImplementedError to maintain compatibility with 
                # environments that do not define these internal success-checking methods.
                try:
                    success = env._get_success()
                    if isinstance(success, torch.Tensor):
                        success_reached = bool(success.any().item())
                    else:
                        success_reached = bool(success)

                    if success_reached:
                        success_flag = True
                        extra_steps = 50
                        logger.info(
                            f"Evaluation episode {i + 1}/{args.num_episodes} reached success at step {st + 1}."
                        )
                except (AttributeError, NotImplementedError):
                    # Gracefully skip if the environment doesn't support automatic success detection
                    pass

            # Attempt to fetch rewards for metric tracking.
            # If the environment doesn't expose a reward method, we fallback to 0.0 
            # to ensure the evaluation loop remains functional across different tasks.
            try:
                reward_value = env._get_rewards()
                if reward_value is None:
                    reward = 0.0
                elif isinstance(reward_value, torch.Tensor):
                    reward = reward_value.item()
                else:
                    reward = float(reward_value)
            except (AttributeError, NotImplementedError, TypeError, ValueError):
                reward = 0.0

            # Accumulate reward for all steps (including post-success steps)
            episode_return += reward
            # Only count length before success (for consistency with episode termination)
            if not success_flag:
                episode_length += 1

            # Update Observation
            observation_dict = env._get_observations()

            # Recording
            if args.save_datasets:
                frame = {
                    k: v
                    for k, v in observation_dict.items()
                    if k != "observation.top_depth"
                }
                frame["task"] = args.task_description
                eval_dataset.add_frame(frame)

            if args.save_video:
                for key, val in observation_dict.items():
                    if "images" in key:
                        episode_frames[key].append(val.copy())

            if success_flag:
                extra_steps -= 1
                if extra_steps <= 0:
                    break

        # --- End of Episode Handling ---
        if success_flag:
            if isinstance(success, torch.Tensor):
                is_success = bool(success.any().item())
            else:
                is_success = bool(success)
        else:
            is_success = False

        if is_success:
            total_success_count += 1

        # Save Datasets
        if args.save_datasets:
            if is_success:
                eval_dataset.save_episode()
                if json_path:
                    variant_name = getattr(env.cfg, "variant_name", "default")
                    append_episode_initial_pose(
                        json_path,
                        episode_index,
                        object_initial_pose,
                        variant_name=variant_name,
                    )
                episode_index += 1
            else:
                eval_dataset.clear_episode_buffer()

        # Save Videos (Using generic util)
        if args.save_video:
            save_videos_from_observations(
                episode_frames,
                success=success if success_flag else torch.tensor(False),
                save_dir=args.video_dir,
                episode_idx=i,
            )

        # Log Metrics
        all_episode_metrics.append(
            {"return": episode_return, "length": episode_length, "success": is_success}
        )
        logger.info(
            f"Episode {i + 1}/{args.num_episodes}: Return={episode_return:.2f}, Length={episode_length}, Success={is_success}"
        )
        logger.info(
            f"Running success rate: {total_success_count}/{i + 1} = {total_success_count / (i + 1):.2%}"
        )

    return all_episode_metrics


def eval(args: argparse.Namespace, simulation_app: Any) -> None:
    """Main entry point for evaluation logic."""

    # 1. Environment Configuration
    env_cfg = parse_env_cfg(args.task, device=args.device)
    if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "use_fabric"):
        env_cfg.sim.use_fabric = False

    # Apply CLI seed settings for environment initialization.
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

    # 2. Initialize Policy
    logger.info(f"Initializing Policy Type: {args.policy_type}")
    if not PolicyRegistry.is_registered(args.policy_type):
        raise ValueError(f"Policy type '{args.policy_type}' not found.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Dynamic bimanual detection
    # We briefly create the env if needed to check num_actions, or use task name heuristic
    is_bimanual = "Bi" in args.task or "bi" in args.task.lower()

    # Create policy instance from registry with appropriate arguments
    # Different policies may require different initialization arguments
    policy_kwargs = {"device": device}

    if args.policy_type == "lerobot":
        # LeRobot policy requires policy path and dataset root for initialization
        if not args.policy_path or not args.dataset_root:
            raise ValueError("--policy_path and --dataset_root are required for lerobot.")
        policy_kwargs.update({
            "policy_path": args.policy_path,
            "dataset_root": args.dataset_root,
            "task_description": args.task_description,
        })
    else:
        # For custom policies, pass policy_path as model_path if provided
        if args.policy_path:
            policy_kwargs["model_path"] = args.policy_path

    # Create policy from registry
    policy = PolicyRegistry.create(args.policy_type, **policy_kwargs)
    logger.info(f"Policy '{args.policy_type}' loaded successfully")

    # 3. Initialize IK Solver（If needed）
    ee_solver = None
    if args.use_ee_pose:
        from lehome.utils import RobotKinematics
        joint_names = [
            "shoulder_pan", 
            "shoulder_lift", 
            "elbow_flex", 
            "wrist_flex", 
            "wrist_roll"
        ]
        ee_solver = RobotKinematics(
            str(args.ee_urdf_path), 
            target_frame_name="gripper_frame_link", 
            joint_names=joint_names
        )
        logger.info(f"IK solver loaded.")
    
    # 4. Create Environment and Run Evaluation
    # Each of the 6 fixed task scenes is specified directly via --task.
    # No variant scheduling needed.
    all_metrics = []

    env = gym.make(args.task, cfg=env_cfg).unwrapped
    try:
        if hasattr(env, "initialize_obs"):
            env.initialize_obs()

        metrics = run_evaluation_loop(
            env=env,
            policy=policy,
            args=args,
            ee_solver=ee_solver,
            is_bimanual=is_bimanual,
        )
        all_metrics.append({"task": args.task, "metrics": metrics})

    finally:
        env.close()

    # 5. Summary
    logger.info("=" * 60)
    logger.info("Evaluation Summary")
    logger.info("=" * 60)

    all_episodes = []
    for entry in all_metrics:
        for m in entry["metrics"]:
            all_episodes.append(m)

    if all_episodes:
        calculate_and_print_metrics(all_episodes)
    else:
        logger.warning("No metrics collected.")
    logger.info("=" * 60)
