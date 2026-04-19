from __future__ import annotations
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg
from isaaclab.envs import ViewerCfg
from isaaclab.sensors import TiledCameraCfg

from lehome.assets.robots.lerobot import SO101_FOLLOWER_CFG, SO101_KINFE_CFG
from ..base.base_env_cfg import BaseEnvCfg
import os


@configclass
class LoftWipeEnvCfg(BaseEnvCfg):
    variant_name: str = "default"
    """Environment configuration inheriting from the base LeHome scene config."""

    # Single-arm task: override base class bimanual defaults (12) with correct single-arm dims
    action_space = 6
    observation_space = 6

    light_intensity: float = 50000.0

    render_cfg: sim_utils.RenderCfg = sim_utils.RenderCfg(rendering_mode="quality", antialiasing_mode="FXAA")
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, render_interval=1, render=render_cfg, use_fabric=False
    )

    robot: ArticulationCfg = SO101_FOLLOWER_CFG.replace(
        prim_path="/World/Robot",
        init_state=SO101_FOLLOWER_CFG.init_state.replace(
            pos=(-2.9185, 2.34977, -0.03197),
            rot=(0.707, 0.0, 0.0, -0.707),
            joint_pos={
                "shoulder_pan": -0.0363,
                "shoulder_lift": -1.7135,
                "elbow_flex": 1.4979,
                "wrist_flex": 1.0534,
                "wrist_roll": -0.085,
                "gripper": -0.01176,
            },
        ),
    )
    wrist_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/gripper/wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.001, 0.1, -0.04),
            rot=(-0.404379, -0.912179, -0.0451242, 0.0486914),
            convention="ros",
        ),  # wxyz
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=36.5,
            focus_distance=400.0,
            horizontal_aperture=36.83,  # For a 75° FOV (assuming square image)
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,  # 30FPS
    )
    top_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/base/top_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, -0.5, 0.6),
            rot=(0.1650476, -0.9862856, 0.0, 0.0),
            convention="ros",
        ),  # wxyz
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.7,
            focus_distance=400.0,
            horizontal_aperture=38.11,  # For a 78° FOV (assuming square image)
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,  # 30FPS
    )

    scene_deactivation_enabled: bool = True
    scene_deactivation_root_path: str = "/World/Scene"
    scene_keep_prim_path_prefixes: tuple[str, ...] = (
        "/World/Scene/Ceiling",
        "/World/Scene/Wall",
        "/World/Scene/Washroom/Toilet033",
    )
    scene_deactivation_log_limit: int = 8

    viewer: ViewerCfg = ViewerCfg(eye=(-2.65, 2.35, 0.7), lookat=(-3.2, 2.35, 0.0))
