from __future__ import annotations
import torch
from typing import Any, Sequence
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import TiledCamera
from pxr import Sdf, UsdGeom, Usd, Gf
from ..base.base_env_cfg import BaseEnvCfg
from .loft_cut_bi_cfg import LoftCutEnvCfg
from ..base.base_env import BaseEnv
from lehome.utils.success_checker import success_checker_cut
from lehome.devices.action_process import preprocess_device_action
from lehome.utils.cutMeshNode import cutMeshNode
from lehome.utils.collision_checker import Collision_Checker
from types import SimpleNamespace
import omni
from isaacsim.core.utils.prims import delete_prim
import transforms3d as t3
import numpy as np
from numpy.random import default_rng
from lehome.utils.random_position import randomize_pose
import os


class LoftCutEnv(BaseEnv):
    """Kitchen cut task built on top of the shared LeHome base environment."""

    cfg: BaseEnvCfg | LoftCutEnvCfg

    def __init__(
        self, cfg: BaseEnvCfg | LoftCutEnvCfg, render_mode: str | None = None, **kwargs
    ):
        self.base_t = tuple(cfg.sausage_init_pos)
        self.base_q_wxyz = tuple(cfg.sausage_init_rot)
        self.sausage_root_path = cfg.sausage_root_prim_path
        self.sausage_mesh_path = f"{self.sausage_root_path}/{cfg.sausage_mesh_prim_name}"
        self.sausage_trigger_path = (
            f"{self.sausage_root_path}/{cfg.sausage_trigger_rel_path}"
        )
        self.sausage_usd_path = cfg.sausage_usd_path
        self.current_sausage_pose = {}  # Will be populated in _setup_scene
        self.stage = omni.usd.get_context().get_stage()
        super().__init__(cfg, render_mode, **kwargs)
        # Additional initialization specific to this environment
        self.action_scale = self.cfg.action_scale
        self.left_joint_pos = self.left_arm.data.joint_pos
        self.right_joint_pos = self.right_arm.data.joint_pos
        # Assume that there is a prim path in the stage
        self.dummy_db = SimpleNamespace()
        self.dummy_db.inputs = SimpleNamespace()
        self.dummy_db.inputs.cut_mesh_path = self.sausage_mesh_path
        self.dummy_db.inputs.knife_mesh_path = (
            "/World/Robot/Right_Robot/gripper/Knife/Knife/Cube"
        )
        self.dummy_db.internal_state = cutMeshNode.internal_state()
        self.dummy_db.inputs.cutEventIn = False
        self.collision_checker = Collision_Checker(
            stage=self.stage, prim_path0=self.sausage_trigger_path
        )
        self.last_if_collision = False

    def _spawn_sausage(self, translation, orientation) -> None:
        cfg = sim_utils.UsdFileCfg(usd_path=self.sausage_usd_path)
        cfg.func(
            self.sausage_root_path,
            cfg,
            translation=translation,
            orientation=orientation,
        )

    def _setup_scene(self):
        """Setup the scene by calling parent method and adding additional assets."""
        # Call parent setup to load the shared LeHome scene and lighting first.
        super()._setup_scene()
        self.left_arm = Articulation(self.cfg.left_robot)
        self.right_arm = Articulation(self.cfg.right_robot)
        self.top_camera = TiledCamera(self.cfg.top_camera)

        self.left_camera = TiledCamera(self.cfg.left_wrist)
        self.right_camera = TiledCamera(self.cfg.right_wrist)
        self.chopping_block = RigidObject(self.cfg.chopping_block)
        self._spawn_sausage(self.base_t, self.base_q_wxyz)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # Store initial local spawn pose relative to the sausage root prim.
        self.current_sausage_pose = {
            "trans": list(self.base_t),
            "rot": list(self.base_q_wxyz),
        }

        # add articulation to scene
        self.scene.articulations["left_arm"] = self.left_arm
        self.scene.articulations["right_arm"] = self.right_arm
        self.scene.rigid_objects["chopping_block"] = self.chopping_block
        self.scene.sensors["top_camera"] = self.top_camera
        self.scene.sensors["left_camera"] = self.left_camera
        self.scene.sensors["right_camera"] = self.right_camera

    def _apply_action(self) -> None:

        if_collision, _, _ = self.collision_checker.meshes_aabb_collide()
        if if_collision == self.last_if_collision and if_collision:
            if_collision = False
        else:
            self.last_if_collision = if_collision

        self.left_arm.set_joint_position_target(self.actions[:, :6])
        self.right_arm.set_joint_position_target(self.actions[:, 6:])
        self.dummy_db.inputs.cutEventIn = if_collision
        cutMeshNode.compute(self.dummy_db)

    def _get_observations(self) -> dict:
        action = self.actions.squeeze(0)
        left_joint_pos = torch.cat(
            [self.left_joint_pos[:, i].unsqueeze(1) for i in range(6)], dim=-1
        )
        right_joint_pos = torch.cat(
            [self.right_joint_pos[:, i].unsqueeze(1) for i in range(6)], dim=-1
        )
        joint_pos = torch.cat([left_joint_pos, right_joint_pos], dim=1)
        joint_pos = joint_pos.squeeze(0)
        top_camera_rgb = self.top_camera.data.output["rgb"]
        top_camera_depth = self.top_camera.data.output["depth"].squeeze()
        depth_mm = self._depth_to_uint16_mm(top_camera_depth)
        left_camera_rgb = self.left_camera.data.output["rgb"]
        right_camera_rgb = self.right_camera.data.output["rgb"]
        observations = {
            "action": action.cpu().detach().numpy(),
            "observation.state": joint_pos.cpu().detach().numpy(),
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
        sausage_count = 0
        sausage_prim = self.stage.GetPrimAtPath(self.sausage_mesh_path)
        sausage_count = len(sausage_prim.GetChildren())
        success = success_checker_cut(sausage_count)
        if isinstance(success, bool):
            success_tensor = torch.tensor(
                [success] * len(self.episode_length_buf), device=self.device
            )
        else:
            success_tensor = torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
        episode_success = success_tensor
        return episode_success

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.left_arm._ALL_INDICES
        super()._reset_idx(env_ids)

        left_joint_pos = self.left_arm.data.default_joint_pos[env_ids]
        right_joint_pos = self.right_arm.data.default_joint_pos[env_ids]
        chopping_block_pose = self.chopping_block.data.default_root_state[env_ids, :7].clone()
        self.left_arm.write_joint_position_to_sim(left_joint_pos, joint_ids=None, env_ids=env_ids)
        self.right_arm.write_joint_position_to_sim(right_joint_pos, joint_ids=None, env_ids=env_ids)
        self.chopping_block.write_root_pose_to_sim(chopping_block_pose, env_ids=env_ids)
        self.object_reset(self.sausage_root_path)

    def preprocess_device_action(
        self, action: dict[str, Any], teleop_device
    ) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)

    def _store_world_pose(self, t_local, q_local_wxyz):
        """Store the local spawn pose directly — no matrix math needed."""
        self.current_sausage_pose = {
            "trans": [float(x) for x in t_local],
            "rot": [float(x) for x in q_local_wxyz],
        }

    def object_reset(self, sausage_path=""):
        delete_prim(sausage_path)
        t_new, q_new = randomize_pose(
            base_translation=self.base_t,
            base_quat_wxyz=self.base_q_wxyz,
            trans_range={
                "x": tuple(self.cfg.sausage_random_x_range),
                "y": tuple(self.cfg.sausage_random_y_range),
                "z": tuple(self.cfg.sausage_random_z_range),
            },
            axis=self.cfg.sausage_random_axis,
            deg_range=self.cfg.sausage_random_deg_range,
            axis_space=self.cfg.sausage_random_axis_space,
            rng=default_rng(),
        )
        self._spawn_sausage(t_new, q_new)
        # Store local spawn pose directly
        self._store_world_pose(t_new, q_new)

    def initialize_obs(self):
        pass

    def get_all_pose(self):
        # Return the stored spawn-point pose instead of current matrix to ensure 
        # that replay starts from the exact same physical initial condition.
        return {"Sausage": self.current_sausage_pose}

    def set_all_pose(self, pose_dict: dict, env_ids: Sequence[int] | None = None):
        """Restore sausage to the recorded local pose.
        Uses xformOp:translate + xformOp:orient to match standardize_xform_ops
        (which is what cfg.func uses when spawning the sausage).
        """
        if "Sausage" not in pose_dict:
            return
        data = pose_dict["Sausage"]
        prim = self.stage.GetPrimAtPath(self.sausage_root_path)
        if not prim.IsValid():
            return
        t = data["trans"]   # local coords under the sausage root prim
        q = data["rot"]     # wxyz quaternion, local

        xformable = UsdGeom.Xformable(prim)
        # Write translate op (xformOp:translate) — same as standardize_xform_ops creates
        translate_attr = prim.GetAttribute("xformOp:translate")
        if translate_attr:
            translate_attr.Set(Gf.Vec3d(float(t[0]), float(t[1]), float(t[2])))
        else:
            op = xformable.AddXformOp(UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.PrecisionDouble)
            op.Set(Gf.Vec3d(float(t[0]), float(t[1]), float(t[2])))

        # Write orient op (xformOp:orient) — same as standardize_xform_ops creates
        orient_attr = prim.GetAttribute("xformOp:orient")
        if orient_attr:
            orient_attr.Set(Gf.Quatd(float(q[0]), float(q[1]), float(q[2]), float(q[3])))
        else:
            op = xformable.AddXformOp(UsdGeom.XformOp.TypeOrient, UsdGeom.XformOp.PrecisionDouble)
            op.Set(Gf.Quatd(float(q[0]), float(q[1]), float(q[2]), float(q[3])))
