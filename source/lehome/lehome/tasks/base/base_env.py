from __future__ import annotations
import torch
import numpy as np
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from lehome.utils.scene_deactivation import deactivate_subtrees_below_root

from .base_env_cfg import BaseEnvCfg


class BaseEnv(DirectRLEnv):
    """Base environment class for the shared LeHome house scene."""

    cfg: BaseEnvCfg

    def __init__(self, cfg: BaseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.action_scale = self.cfg.action_scale

    def _setup_scene(self):
        """Set up the shared LeHome house scene and task-independent lighting."""

        # Load the shared LeHome house scene
        loft = sim_utils.UsdFileCfg(usd_path=self.cfg.path_scene)
        loft.func(
            "/World/Scene",
            loft,
            translation=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
        )
        self._deactivate_scene_prims()

        light_cfg = sim_utils.DomeLightCfg(
            intensity=self.cfg.light_intensity, color=(0.75, 0.75, 0.75)
        )
        light_cfg.func("/World/Light", light_cfg)

    def _deactivate_scene_prims(self) -> None:
        if not getattr(self.cfg, "scene_deactivation_enabled", False):
            return

        if not self.cfg.scene_keep_prim_path_prefixes:
            print(
                f"[{self.__class__.__name__}][SceneDeactivate] "
                "No whitelist configured. Non-whitelisted subtrees under "
                f"{self.cfg.scene_deactivation_root_path} will be deactivated."
            )

        stats = deactivate_subtrees_below_root(
            stage=self.scene.stage,
            root_path=self.cfg.scene_deactivation_root_path,
            keep_prim_path_prefixes=self.cfg.scene_keep_prim_path_prefixes,
        )
        self.scene_deactivation_stats = stats

        if not stats["root_found"]:
            print(
                f"[{self.__class__.__name__}][SceneDeactivate] "
                f"Root prim not found: {self.cfg.scene_deactivation_root_path}"
            )
            return

        print(
            f"[{self.__class__.__name__}][SceneDeactivate] "
            f"processed={stats['processed_count']}, "
            f"deactivated={stats['deactivated_count']}, "
            f"kept={stats['kept_count']}, "
            f"bridges={stats['bridge_count']}, "
            f"already_inactive={stats['already_inactive_count']}"
        )

        log_limit = max(int(self.cfg.scene_deactivation_log_limit), 0)
        if log_limit == 0:
            return

        deactivated_preview = stats["deactivated_paths"][:log_limit]
        if deactivated_preview:
            print(
                f"[{self.__class__.__name__}][SceneDeactivate] deactivated sample: "
                + ", ".join(deactivated_preview)
            )

        kept_preview = stats["kept_paths"][:log_limit]
        if kept_preview:
            print(
                f"[{self.__class__.__name__}][SceneDeactivate] kept sample: "
                + ", ".join(kept_preview)
            )

        bridge_preview = stats["bridge_paths"][:log_limit]
        if bridge_preview:
            print(
                f"[{self.__class__.__name__}][SceneDeactivate] bridge sample: "
                + ", ".join(bridge_preview)
            )

        already_inactive_preview = stats["already_inactive_paths"][:log_limit]
        if already_inactive_preview:
            print(
                f"[{self.__class__.__name__}][SceneDeactivate] already inactive sample: "
                + ", ".join(already_inactive_preview)
            )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Pre-physics step processing."""
        self.actions = self.action_scale * actions.clone()
        # step counter increments once per control step

    @staticmethod
    def _depth_to_uint16_mm(depth_tensor: torch.Tensor) -> np.ndarray:
        """Convert Isaac Sim depth in meters to millimeters stored as uint16."""
        depth_np = depth_tensor.detach().cpu().numpy().copy()
        return np.clip(depth_np * 1000.0, 0, 65535).astype(np.uint16)

    def _apply_action(self) -> None:
        """Apply actions to the robot."""
        pass

    def _get_observations(self) -> dict:
        """Get observations from the environment."""
        pass

    def _get_rewards(self) -> torch.Tensor:
        pass

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        pass
