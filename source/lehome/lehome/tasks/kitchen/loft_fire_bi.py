from __future__ import annotations
import torch
from typing import Any, Sequence

from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import TiledCamera
from .loft_fire_bi_cfg import LoftFireEnvCfg
from ..base.base_env import BaseEnv
from ..base.base_env_cfg import BaseEnvCfg
from lehome.devices.action_process import preprocess_device_action

class LoftFireEnv(BaseEnv):
    """Kitchen fire task built on top of the shared LeHome base environment."""

    KNOB_NAMES = (
        "knob_front_center_joint",
        "knob_front_left_joint",
        "knob_front_right_joint",
        "knob_rear_center_joint",
    )

    cfg: BaseEnvCfg | LoftFireEnvCfg

    def __init__(
        self, cfg: BaseEnvCfg | LoftFireEnvCfg, render_mode: str | None = None, **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self.action_scale = self.cfg.action_scale
        self.left_joint_pos = self.left_arm.data.joint_pos
        self.right_joint_pos = self.right_arm.data.joint_pos
        self.knob_joint_ids, _ = self.stovetop.find_joints(list(self.KNOB_NAMES), preserve_order=True)

        # Programmatically enable the Flow extension for fire effects
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("omni.flowusd.bundle")
        
    def _setup_scene(self):
        """Setup the scene and additional assets."""
        super()._setup_scene()
        
        # Setup Robots
        self.left_arm = Articulation(self.cfg.left_robot)
        self.right_arm = Articulation(self.cfg.right_robot)
        self.stovetop = Articulation(
            ArticulationCfg(
                prim_path="/World/Scene/Kitchen/Stovetop017",
                actuators={
                    "knobs": ImplicitActuatorCfg(
                        joint_names_expr=["knob_.*_joint"],
                        effort_limit_sim=40.0,
                        velocity_limit_sim=20.0,
                        stiffness=0.0,
                        damping=0.0,
                    )
                },
            )
        )
        
        # Setup Cameras
        self.left_wrist = TiledCamera(self.cfg.left_wrist)
        self.right_wrist = TiledCamera(self.cfg.right_wrist)
        self.top_camera = TiledCamera(self.cfg.top_camera)

        # Register to scene
        self.scene.articulations["left_arm"] = self.left_arm
        self.scene.articulations["right_arm"] = self.right_arm
        self.scene.articulations["stovetop"] = self.stovetop

        self.scene.sensors["left_wrist"] = self.left_wrist
        self.scene.sensors["right_wrist"] = self.right_wrist
        self.scene.sensors["top_camera"] = self.top_camera

    def initialize_obs(self):
        pass

    def _apply_action(self) -> None:
        # Provide actions to robots
        left_actions = self.actions[:, 0:6]
        right_actions = self.actions[:, 6:12]

        self.left_arm.set_joint_position_target(left_actions)
        self.right_arm.set_joint_position_target(right_actions)

    def _get_observations(self) -> dict:
        action = self.actions.squeeze(0)

        # Left and right joint positions
        left_joint_pos = torch.cat(
            [self.left_joint_pos[:, i].unsqueeze(1) for i in range(6)], dim=-1
        ).squeeze(0)
        right_joint_pos = torch.cat(
            [self.right_joint_pos[:, i].unsqueeze(1) for i in range(6)], dim=-1
        ).squeeze(0)

        top_camera_rgb = self.top_camera.data.output["rgb"]
        top_camera_depth = self.top_camera.data.output["depth"].squeeze()
        depth_mm = self._depth_to_uint16_mm(top_camera_depth)
        left_camera_rgb = self.left_wrist.data.output["rgb"]
        right_camera_rgb = self.right_wrist.data.output["rgb"]

        observations = {
            "action": action.cpu().detach().numpy(),
            "observation.state": torch.cat((left_joint_pos, right_joint_pos)).cpu().detach().numpy(),
            "observation.images.top_rgb": top_camera_rgb.cpu()
            .detach()
            .numpy()
            .squeeze(),
            "observation.images.left_rgb": left_camera_rgb.cpu()
            .detach()
            .numpy()
            .squeeze(),
            "observation.images.right_rgb": right_camera_rgb.cpu()
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
        success = torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
        return success

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.left_arm._ALL_INDICES
        super()._reset_idx(env_ids)

        left_joint_pos = self.left_arm.data.default_joint_pos[env_ids]
        right_joint_pos = self.right_arm.data.default_joint_pos[env_ids]

        self.left_arm.write_joint_position_to_sim(
            left_joint_pos, joint_ids=None, env_ids=env_ids
        )
        self.right_arm.write_joint_position_to_sim(
            right_joint_pos, joint_ids=None, env_ids=env_ids
        )

        if len(self.knob_joint_ids) > 0:
            knob_reset_pos = torch.zeros(
                (len(env_ids), len(self.knob_joint_ids)),
                device=self.device,
                dtype=left_joint_pos.dtype,
            )
            knob_reset_vel = torch.zeros_like(knob_reset_pos)
            self.stovetop.write_joint_position_to_sim(
                knob_reset_pos, joint_ids=self.knob_joint_ids, env_ids=env_ids
            )
            self.stovetop.write_joint_velocity_to_sim(
                knob_reset_vel, joint_ids=self.knob_joint_ids, env_ids=env_ids
            )
            self.stovetop.set_joint_position_target(
                knob_reset_pos, joint_ids=self.knob_joint_ids, env_ids=env_ids
            )

    def preprocess_device_action(
        self, action: dict[str, Any], teleop_device
    ) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)

    def get_all_pose(self):
        return {}

    def set_all_pose(self, pose_dict, env_ids: Sequence[int] | None = None):
        return None
