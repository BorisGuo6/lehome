from __future__ import annotations
import isaaclab.sim as sim_utils
from isaaclab.envs import ViewerCfg
from isaaclab.sim import SimulationCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg

from lehome.assets.robots.lerobot import SO101_FOLLOWER_CFG, SO101_KINFE_CFG
from ..base.base_env_cfg import BaseEnvCfg

import os


@configclass
class LoftCutEnvCfg(BaseEnvCfg):
    variant_name: str = "default"
    """Environment configuration inheriting from the base LeHome scene config."""

    light_intensity: float = 50000.0

    render_cfg: sim_utils.RenderCfg = sim_utils.RenderCfg(rendering_mode="quality", antialiasing_mode="FXAA")
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, render_interval=1, render=render_cfg, use_fabric=False
    )

    left_robot: ArticulationCfg = SO101_FOLLOWER_CFG.replace(
        prim_path="/World/Robot/Left_Robot",
        init_state=SO101_FOLLOWER_CFG.init_state.replace(
            pos=(-3.49, 6.5281, 0.768), 
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
    right_robot: ArticulationCfg = SO101_KINFE_CFG.replace(
        prim_path="/World/Robot/Right_Robot",
        init_state=SO101_KINFE_CFG.init_state.replace(
            pos=(-3.49, 6.8781, 0.768), 
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
    right_wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/Right_Robot/gripper/right_wrist_camera",
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
        prim_path="/World/Robot/Right_Robot/base/top_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.225, -0.5, 0.6),
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
    chopping_block: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Cut/chopping_block",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.getcwd()
            + "/Assets/scenes/1BRAPT_LeHome/Assets/ChoppingBlock/ChoppingBlock.usd"
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-3.75, 6.7, 0.81),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )
    sausage_root_prim_path: str = "/World/Cut/Sausage001"
    sausage_mesh_prim_name: str = "Sausage001"
    sausage_trigger_rel_path: str = "Trigger/Cube"
    sausage_usd_path: str = (
        os.getcwd() + "/Assets/objects/Volumetric_Objects/Sausage001/Sausage001.usd"
    )
    sausage_init_pos: tuple[float, float, float] = (-3.75, 6.7, 0.84059)
    sausage_init_rot: tuple[float, float, float, float] = (-0.23287, -0.02628, 0.02471, 0.97184)
    sausage_random_x_range: tuple[float, float] = (-0.04, 0.04)
    sausage_random_y_range: tuple[float, float] = (-0.04, 0.04)
    sausage_random_z_range: tuple[float, float] = (0.0, 0.0)
    sausage_random_axis: str = "z"
    sausage_random_axis_space: str = "local"
    sausage_random_deg_range: float = 20.0

    scene_deactivation_enabled: bool = True
    scene_deactivation_root_path: str = "/World/Scene"
    scene_keep_prim_path_prefixes: tuple[str, ...] = (
        "/World/Scene/Ceiling",
        "/World/Scene/Wall",
        "/World/Scene/Kitchen/Kitchen_Cabinet002",
        "/World/Scene/Kitchen/Stovetop017",
        "/World/Scene/Kitchen/SeasoningBox001",
        "/World/Scene/Kitchen/Jar046",
        "/World/Scene/Kitchen/Shovel008",
        "/World/Scene/Kitchen/Shovel007",
        "/World/Scene/Kitchen/WoodenSpoon013",
    )
    scene_deactivation_log_limit: int = 8

    viewer: ViewerCfg = ViewerCfg(eye=(-3.2, 6.7, 1.5), lookat=(-3.6, 6.7, 1.0))
