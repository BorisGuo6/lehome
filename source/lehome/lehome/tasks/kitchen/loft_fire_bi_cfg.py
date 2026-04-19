from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.envs import ViewerCfg
from isaaclab.sim import SimulationCfg
import os

from lehome.assets.robots.lerobot import SO101_FOLLOWER_CFG
from ..base.base_env_cfg import BaseEnvCfg

@configclass
class LoftFireEnvCfg(BaseEnvCfg):
    variant_name: str = "default"
    """Environment configuration inheriting from the base LeHome scene config."""

    render_cfg: sim_utils.RenderCfg = sim_utils.RenderCfg(rendering_mode="quality", antialiasing_mode="FXAA")
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, render_interval=1, render=render_cfg, use_fabric=False
    )

    # robot
    left_robot: ArticulationCfg = SO101_FOLLOWER_CFG.replace(
        prim_path="/World/Robot/Left_Robot",
        init_state=SO101_FOLLOWER_CFG.init_state.replace(
            pos=(-3.4, 5.7, 0.78),  
            rot=(0.707, 0.0, 0.0, -0.707),
            joint_pos={
                "shoulder_pan": -1.2363,
                "shoulder_lift": -1.7135,
                "elbow_flex": 1.4979,
                "wrist_flex": 1.0534,
                "wrist_roll": -0.085,
                "gripper": -0.01176,
            },
        ), 
    )
    right_robot: ArticulationCfg = SO101_FOLLOWER_CFG.replace(
        prim_path="/World/Robot/Right_Robot",
        init_state=SO101_FOLLOWER_CFG.init_state.replace(
            pos=(-3.4, 6.2, 0.78), 
            rot=(0.707, 0.0, 0.0, -0.707),
            joint_pos={
                "shoulder_pan": 1.2363,
                "shoulder_lift": -1.7135,
                "elbow_flex": 1.4979,
                "wrist_flex": 1.0534,
                "wrist_roll": 0.085,
                "gripper": -0.01176,
            },
        ),
    )
    
    left_wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/Left_Robot/gripper/left_wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.001, 0.1, -0.04),
            rot=(-0.404379, -0.912179, -0.0451242, 0.0486914),
            convention="ros",
        ),  
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=36.5,
            focus_distance=400.0,
            horizontal_aperture=36.83,  
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,  
    )
    
    right_wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/Right_Robot/gripper/right_wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.001, 0.1, -0.04),
            rot=(-0.404379, -0.912179, -0.0451242, 0.0486914),
            convention="ros",
        ),  
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=36.5,
            focus_distance=400.0,
            horizontal_aperture=36.83,  
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,  
    )
    
    top_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/Right_Robot/base/top_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.225, -0.72, 0.6),
            rot=(0.2588190, -0.9659258, 0.0, 0.0),
            convention="ros",
        ),  
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=25.7,
            focus_distance=400.0,
            horizontal_aperture=38.11,  
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
        "/World/Scene/Kitchen/Kitchen_Cabinet002",
        "/World/Scene/Kitchen/Stovetop017",
        "/World/Scene/Kitchen/ToasterOven009",
    )
    scene_deactivation_log_limit: int = 8

    viewer: ViewerCfg = ViewerCfg(eye=(-3.0, 5.95, 1.6), lookat=(-3.5, 5.95, 1.0))
