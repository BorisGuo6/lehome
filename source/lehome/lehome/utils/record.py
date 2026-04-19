"""Utilities for dataset recording and replay management."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Try to import OmegaConf for handling ListConfig/DictConfig
try:
    from omegaconf import DictConfig, ListConfig, OmegaConf

    HAS_OMEGACONF = True
except ImportError:
    HAS_OMEGACONF = False
    OmegaConf = None


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz: int):
        """Initialize rate limiter.

        Args:
            hz: Frequency to enforce in Hz.
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env) -> None:
        """Attempt to sleep at the specified rate in Hz.

        Args:
            env: Environment instance with a sim.render() method.
        """
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # Detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def get_next_experiment_path_with_gap(base_path: Path) -> Path:
    """Find the first available numbered subdirectory under base_path.

    Args:
        base_path: Base directory to search for available indices.

    Returns:
        Path to the next available numbered subdirectory (e.g. base_path/003).
    """
    base_path.mkdir(parents=True, exist_ok=True)

    # Collect existing indices
    indices = set()
    for folder in base_path.iterdir():
        if folder.is_dir():
            try:
                indices.add(int(folder.name))
            except ValueError:
                continue

    # Find the first available index
    folder_index = 1
    while folder_index in indices:
        folder_index += 1

    return base_path / f"{folder_index:03d}"


def _ndarray_to_list(obj: Any) -> Any:
    """Recursively convert numpy arrays and OmegaConf objects to JSON-serializable types.

    Args:
        obj: Object to convert (numpy array, dict, list, or OmegaConf type).

    Returns:
        JSON-serializable Python object.
    """
    # Handle OmegaConf types first (before checking for dict/list)
    if HAS_OMEGACONF and isinstance(obj, (ListConfig, DictConfig)):
        obj = OmegaConf.to_container(obj, resolve=True)

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_ndarray_to_list(x) for x in obj]
    else:
        return obj


def append_episode_initial_pose(
    json_path: Any,
    episode_idx: int,
    object_initial_pose: Any,
    variant_name: Optional[str] = None,
) -> None:
    """Append initial pose information to the JSON file with hierarchical structure.

    Saves the initial pose of objects after each environment reset in a nested
    JSON format for easy alignment and lookup.

    Args:
        json_path: Path to the JSON file.
        episode_idx: Episode index.
        object_initial_pose: Initial pose from env.get_all_pose() after reset.
            Typically a dict keyed by task-specific object names, for example
            ``{"Garment": {"trans": ..., "rot": ...}}`` or
            ``{"Cup": {"trans": ..., "rot": ...}}``.
        variant_name: Optional variant or object name (e.g. "Top_Long_Seen_0").
    """
    if variant_name is None:
        variant_name = "default"

    if object_initial_pose is None:
        return

    # Store directly, unified dictionary handles structure
    pose_list = object_initial_pose
    pose_list = _ndarray_to_list(pose_list)


    episode_rec: Dict[str, Any] = {"object_initial_pose": pose_list}

    # Read existing data
    json_path = Path(json_path)
    data: Dict[str, Any] = {}

    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as fin:
                data = json.load(fin)
        except (json.JSONDecodeError, FileNotFoundError):
            data = {}

    # Update or create variant entry
    if variant_name not in data:
        data[variant_name] = {}

    data[variant_name][str(episode_idx)] = episode_rec

    # Write back to file
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as fout:
        json.dump(data, fout, indent=2, ensure_ascii=False)
