"""Common utilities for Isaac Sim app management and robot home positions."""

import argparse
import logging
import sys
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnv

# ---------------------------------------------------------------------------
# Robot home positions (joint angles in radians)
# ---------------------------------------------------------------------------

SINGLE_ARM_HOME_POSITION = np.array(
    [
        -0.0363,   # shoulder_pan
        -1.7135,   # shoulder_lift
        1.4979,    # elbow_flex
        1.0534,    # wrist_flex
        -0.085,    # wrist_roll
        -0.01176,  # gripper
    ],
    dtype=np.float32,
)

LEFT_ARM_HOME_POSITION = np.array(
    [
        -1.2363,   # shoulder_pan
        -1.7135,   # shoulder_lift
        1.4979,    # elbow_flex
        1.0534,    # wrist_flex
        -0.085,    # wrist_roll
        -0.01176,  # gripper
    ],
    dtype=np.float32,
)

RIGHT_ARM_HOME_POSITION = np.array(
    [
        1.2363,    # shoulder_pan
        -1.7135,   # shoulder_lift
        1.4979,    # elbow_flex
        1.0534,    # wrist_flex
        -0.085,    # wrist_roll
        -0.01176,  # gripper
    ],
    dtype=np.float32,
)

DUAL_ARM_HOME_POSITION = np.concatenate(
    [LEFT_ARM_HOME_POSITION, RIGHT_ARM_HOME_POSITION]
)


def setup_cli_logging(level: int = logging.INFO) -> None:
    """Configure terminal logging for CLI entry scripts.

    This helper ensures those messages are
    printed to the terminal consistently.

    Args:
        level: Logging level for the root logger.
    """
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


# ---------------------------------------------------------------------------
# Isaac Sim app management
# ---------------------------------------------------------------------------


def launch_app_from_args(args: argparse.Namespace):
    """Launch Isaac Sim app from already parsed arguments.

    Args:
        args: Parsed command-line arguments (must include AppLauncher args).

    Returns:
        SimulationApp instance.
    """
    from isaaclab.app import AppLauncher

    args.kit_args = (
        "--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error"
    )

    if getattr(args, "enable_cameras", False) and not getattr(args, "experience", ""):
        args.experience = (
            "isaaclab.python.headless.rendering.kit"
            if getattr(args, "headless", False)
            else "isaaclab.python.rendering.kit"
        )

    app_launcher = AppLauncher(vars(args))
    return app_launcher.app


def close_app(simulation_app) -> None:
    """Close Isaac Sim app.

    Args:
        simulation_app: The running SimulationApp instance to close.
    """
    simulation_app.close()


# ---------------------------------------------------------------------------
# Environment stabilization
# ---------------------------------------------------------------------------


def stabilize_robot(
    env: "DirectRLEnv",
    num_steps: int = 20,
) -> None:
    """Stabilize the robot and environment after reset by running passive physics steps.

    Moves the robot to its hardcoded home position. This ensures 
    consistent starting conditions across all episodes and tasks.

    Args:
        env: Environment instance.
        num_steps: Number of stabilization steps to run.
    """
    if num_steps <= 0:
        return

    with torch.no_grad():
        # 1. Determine action dimension safely
        try:
            action_dim = env.cfg.action_space
        except AttributeError:
            try:
                initial_obs = env._get_observations()
                action_dim = len(initial_obs["observation.state"])
            except Exception:
                action_dim = 6

        # 2. Select hardcoded home position
        if action_dim == 12:
            home_joints = DUAL_ARM_HOME_POSITION
        elif action_dim == 6:
            home_joints = SINGLE_ARM_HOME_POSITION
        else:
            # Try to get from robot asset as a fallback for non-standard arms
            try:
                if hasattr(env, "robot"):
                    home_joints = env.robot.data.default_joint_pos[0].cpu().numpy()
                else:
                    home_joints = np.zeros(action_dim, dtype=np.float32)
            except:
                home_joints = np.zeros(action_dim, dtype=np.float32)

        home_action = torch.from_numpy(home_joints).float().to(env.device).unsqueeze(0)

        # 3. Step the environment
        for step_idx in range(num_steps):
            env.step(home_action)
            if (step_idx + 1) % 10 == 0 or step_idx == num_steps - 1:
                env.render()
