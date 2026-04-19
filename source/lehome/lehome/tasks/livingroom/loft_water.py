from __future__ import annotations
import os
import torch

# from dataclasses import MISSING
from typing import Any, Dict, List, Sequence

from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import TiledCamera
from .loft_water_cfg import LoftWaterEnvCfg
from ..base.base_env import BaseEnv
from ..base.base_env_cfg import BaseEnvCfg
from lehome.devices.action_process import preprocess_device_action
from omegaconf import OmegaConf
import numpy as np
from lehome.assets.object.fluid import FluidObject
from pxr import Gf, Usd, UsdGeom


class LoftWaterEnv(BaseEnv):
    """Living-room water task built on top of the shared LeHome base environment."""

    cfg: BaseEnvCfg | LoftWaterEnvCfg

    def __init__(
        self,
        cfg: BaseEnvCfg | LoftWaterEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        # Additional initialization specific to this environment

        self.action_scale = self.cfg.action_scale
        self.joint_pos = self.robot.data.joint_pos

    def _setup_scene(self):
        """Setup the scene by calling parent method and adding additional assets."""
        # Call parent setup to load the shared LeHome scene and lighting first.
        super()._setup_scene()
        self.robot = Articulation(self.cfg.robot)
        self.top_camera = TiledCamera(self.cfg.top_camera)
        self.wrist_camera = TiledCamera(self.cfg.wrist_camera)

        self.object = FluidObject(
            env_id=0,
            env_origin=torch.zeros(1, 3),
            prim_path="/World/Object/fluid_items/fluid_items_1",
            usd_path=os.getcwd() + "/Assets/objects/Fluids/water/water.usdc",
            config=OmegaConf.load(
                os.getcwd()
                + "/source/lehome/lehome/tasks/livingroom/config_file/fluid.yaml"
            ),
            use_container=True,
        )
        self.bowl = RigidObject(self.cfg.bowl)
        self.scene.rigid_objects["bowl"] = self.bowl

        # add articulation to scene
        self.scene.articulations["robot"] = self.robot

        self.scene.sensors["top_camera"] = self.top_camera
        self.scene.sensors["wrist_camera"] = self.wrist_camera

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.actions)

    def _get_observations(self) -> dict:
        action = self.actions.squeeze(0)
        joint_pos = torch.cat(
            [self.joint_pos[:, i].unsqueeze(1) for i in range(6)], dim=-1
        ).squeeze(0)

        top_camera_rgb = self.top_camera.data.output["rgb"]
        top_camera_depth = self.top_camera.data.output["depth"].squeeze()
        depth_mm = self._depth_to_uint16_mm(top_camera_depth)
        wrist_camera_rgb = self.wrist_camera.data.output["rgb"]
        observations = {
            "action": action.cpu().detach().numpy(),
            "observation.state": joint_pos.cpu().detach().numpy(),
            "observation.images.top_rgb": top_camera_rgb.cpu()
            .detach()
            .numpy()
            .squeeze(),
            "observation.images.wrist_rgb": wrist_camera_rgb.cpu()
            .detach()
            .numpy()
            .squeeze(),
            "observation.top_depth": depth_mm,
        }
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = torch.zeros_like(self.episode_length_buf, dtype=torch.float32)
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _get_success(self) -> torch.Tensor:
        particle_positions, _, _ = self.object.get_particle_positions(visualize=False)
        if len(particle_positions) == 0:
            return torch.zeros_like(self.episode_length_buf, dtype=torch.bool)

        local_to_world = UsdGeom.Xformable(
            self.object.point_instancer.GetPrim()
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        world_positions = np.asarray(
            [
                tuple(
                    local_to_world.Transform(
                        Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2]))
                    )
                )
                for pos in particle_positions
            ],
            dtype=np.float32,
        )

        bowl_center = self.bowl.data.root_state_w[0, :3].detach().cpu().numpy()
        radial_distance_sq = np.sum(
            (world_positions[:, :2] - bowl_center[:2]) ** 2, axis=1
        )
        inside_bowl = (
            (radial_distance_sq <= 0.09**2)
            & (world_positions[:, 2] >= bowl_center[2] - 0.02)
            & (world_positions[:, 2] <= bowl_center[2] + 0.08)
        )
        success = bool(np.mean(inside_bowl) >= 0.3)
        return torch.full(
            (len(self.episode_length_buf),),
            success,
            device=self.device,
            dtype=torch.bool,
        )

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        self.object.reset()
        bowl_pos = self.bowl.data.default_root_state[env_ids].clone()
        # Create random values on the same device as bowl_pos
        rand_vals_1 = torch.empty(len(env_ids), 2, device=bowl_pos.device).uniform_(-0.05, 0.05)
        # Add random position perturbation to the bowl
        random_bowl_pos = bowl_pos.clone()
        random_bowl_pos[..., :2] += rand_vals_1
        random_bowl_pos[..., 7:] = 0.0  # reset velocity
        self.bowl.write_root_state_to_sim(random_bowl_pos, env_ids=env_ids)
        self.robot.write_joint_position_to_sim(joint_pos, joint_ids=None, env_ids=env_ids)

    def preprocess_device_action(
        self, action: dict[str, Any], teleop_device
    ) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)

    def initialize_obs(self):
        self.object.initialize()
        # RigidObject use reset() to initialize
        self.bowl.reset()

    def get_all_pose(self):
        # Cup: FluidObject - uses get_pose_data() for unified nested dict
        cup_pose = self.object.get_pose_data()  # {"trans":..., "rot":..., "scale":...}
        # Bowl: RigidObject - root_state_w is [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        bowl_root = self.bowl.data.root_state_w[0]
        bowl_pose = {
            "trans": bowl_root[:3].cpu().tolist(),
            "rot":   bowl_root[3:7].cpu().tolist(),  # already quaternion [w,x,y,z]
        }
        return {"Cup": cup_pose, "Bowl": bowl_pose}

    def set_all_pose(self, pose_dict, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = self.bowl._ALL_INDICES
        # Restore Cup via FluidObject
        if "Cup" in pose_dict:
            self.object.set_pose_from_data(pose_dict["Cup"])
        # Restore Bowl via RigidObject root state
        if "Bowl" in pose_dict:
            bowl_data = pose_dict["Bowl"]
            trans = bowl_data["trans"]
            rot   = bowl_data["rot"]
            bowl_root_state = self.bowl.data.default_root_state[env_ids].clone()
            bowl_root_state[..., :3] = torch.tensor(trans, dtype=torch.float32)
            bowl_root_state[..., 3:7] = torch.tensor(rot, dtype=torch.float32)
            bowl_root_state[..., 7:] = 0.0  # zero velocity
            self.bowl.write_root_state_to_sim(bowl_root_state, env_ids=env_ids)
