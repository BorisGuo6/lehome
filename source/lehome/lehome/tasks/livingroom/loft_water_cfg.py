from __future__ import annotations
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg
from isaaclab.envs import ViewerCfg
from isaaclab.sensors import TiledCameraCfg

from lehome.assets.robots.lerobot import SO101_FOLLOWER_CFG, SO101_KINFE_CFG
from ..base.base_env_cfg import BaseEnvCfg
import os


@configclass
class LoftWaterEnvCfg(BaseEnvCfg):
    variant_name: str = "default"
    """Environment configuration inheriting from the base LeHome scene config."""

    # Single-arm task: override base class bimanual defaults (12) with correct single-arm dims
    action_space = 6
    observation_space = 6
    
    light_intensity: float = 12000.0

    render_cfg: sim_utils.RenderCfg = sim_utils.RenderCfg(rendering_mode="quality", antialiasing_mode="FXAA")
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, render_interval=1, render=render_cfg, use_fabric=False
    )

    robot: ArticulationCfg = SO101_FOLLOWER_CFG.replace(
        prim_path="/World/Robot/Robot",
        init_state=SO101_FOLLOWER_CFG.init_state.replace(
            pos=(3, 0.58, 0.76541), 
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
        prim_path="/World/Robot/Robot/gripper/wrist_camera",
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
        prim_path="/World/Robot/Robot/base/top_camera",
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

    bowl: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Object/bowl",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.getcwd() + "/Assets/objects/Fluids/Bowl016/Bowl016.usd"
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(2.7, 0.45, 0.83),
            rot=(0.0, 0.0, 0.0, 1),
        ),
    )

    scene_deactivation_enabled: bool = True
    scene_deactivation_root_path: str = "/World/Scene"
    scene_keep_prim_path_prefixes: tuple[str, ...] = (
        "/World/Scene/Ceiling",
        "/World/Scene/Wall",
        "/World/Scene/Livingroom/Table062",
        "/World/Scene/Livingroom/Chair024",
        "/World/Scene/Livingroom/Chair024_01",
        "/World/Scene/Livingroom/Chair024_02",
        "/World/Scene/Livingroom/Chair024_03",
    )
    scene_deactivation_log_limit: int = 8

    viewer: ViewerCfg = ViewerCfg(eye=(3.4, 0.6, 1.3), lookat=(3.0, 0.58, 1.0))
