# Scene Deactivation Guide

This guide explains the path-prefix-based scene subtree deactivation system implemented in the base environment layer of LeHome.

## Table of Contents

- [Scene Deactivation Guide](#scene-deactivation-guide)
  - [Table of Contents](#table-of-contents)
  - [Important Notice (Read First)](#important-notice-read-first)
  - [1. Overview](#1-overview)
  - [2. What This Feature Does](#2-what-this-feature-does)
  - [3. Why This Helps Performance](#3-why-this-helps-performance)
  - [4. Common Configuration Fields](#4-common-configuration-fields)
  - [5. Task Defaults in This Repository](#5-task-defaults-in-this-repository)
  - [6. Example: Wiping Task](#6-example-wiping-task)
  - [7. Example: Keep an Entire Room](#7-example-keep-an-entire-room)
  - [8. Startup Logs](#8-startup-logs)

## Important Notice (Read First)

Before configuring or modifying scene deactivation behavior, please confirm the following:

1. The current base scene is loaded from: `/Assets/scenes/1BRAPT_LeHome/1BRAPT_LeHome.usd`.
2. The assets under `/Assets/scenes/1BRAPT_LeHome/Assets` are already preloaded into `/Assets/scenes/1BRAPT_LeHome/1BRAPT_LeHome.usd`.
3. Additional assets that are not yet loaded into the base scene are available under `/Assets/scenes/1BRAPT_LeHome/Assets_not_loaded`. If your task requires them, import the needed assets into `/Assets/scenes/1BRAPT_LeHome/1BRAPT_LeHome.usd` before configuring scene deactivation.
4. To avoid incorrect keep/deactivate settings, read the full mechanism described in the rest of this Scene Deactivation Guide before making scene configuration changes.

Recommended reading order:

- first, confirm whether the assets required by your task are already present in the base scene; if not, import the needed assets from `/Assets/scenes/1BRAPT_LeHome/Assets_not_loaded` into `/Assets/scenes/1BRAPT_LeHome/1BRAPT_LeHome.usd`
- then, map those assets to keep-path prefixes
- finally, apply and validate deactivation settings with startup logs

## 1. Overview

Some tasks only need a small part of the house scene. Keeping every room active at the same time can add unnecessary rendering, physics parsing, and collision-processing overhead. To address this, LeHome supports scene subtree deactivation with a path-based whitelist.

The feature is implemented through:

- `source/lehome/lehome/utils/scene_deactivation.py`
- `source/lehome/lehome/tasks/base/base_env.py`
- `source/lehome/lehome/tasks/base/base_env_cfg.py`

## 2. What This Feature Does

The feature is implemented in the base environment, so every task inheriting from `BaseEnv` can use it.

- scene root: `/World/Scene`
- control style: keep-by-path-prefix whitelist
- behavior:
  - any subtree whose path matches the whitelist stays active
  - if a parent node is not whitelisted but one of its descendants is whitelisted, the parent is kept as a bridge node
  - any subtree that does not contain a whitelisted descendant is deactivated with `SetActive(False)`

This means the logic supports room-level paths and object-level paths such as:

- `/World/Scene/Washroom`
- `/World/Scene/Washroom/Toilet033`
- `/World/Scene/Kitchen/Stovetop017`

## 3. Why This Helps Performance

Deactivated subtrees no longer participate as active prims in the scene. In practice, this reduces:

- rendering cost for irrelevant rooms and objects
- physics parsing cost during startup
- collision and rigid-body processing for unrelated assets
- runtime overhead from keeping the full apartment scene active

Important behavior:

- `SetActive(False)` removes both visuals and physics for that subtree
- this is stronger than only disabling collisions
- if an object must remain visible or simulated, it must be included in the whitelist

## 4. Common Configuration Fields

These fields are defined in `BaseEnvCfg`:

```python
scene_deactivation_enabled: bool = False
scene_deactivation_root_path: str = "/World/Scene"
scene_keep_prim_path_prefixes: tuple[str, ...] = ()
scene_deactivation_log_limit: int = 8
```

Meaning:

- `scene_deactivation_enabled`: enable or disable the feature
- `scene_deactivation_root_path`: root prim under which the traversal runs
- `scene_keep_prim_path_prefixes`: tuple of paths to keep active
- `scene_deactivation_log_limit`: number of sample paths shown in logs

## 5. Task Defaults in This Repository

All current task config files enable scene deactivation.

| Task config file | Enabled | Default keep prefixes |
|------------------|---------|-----------------------|
| `source/lehome/lehome/tasks/bedroom/garment_bi_cfg.py` | `True` | `Ceiling`, `Wall`, `Bedroom/Table038_01` |
| `source/lehome/lehome/tasks/bedroom/garment_fling_bi_cfg.py` | `True` | `Ceiling`, `Wall`, `Bedroom/Table038_01` |
| `source/lehome/lehome/tasks/kitchen/loft_burger_bi_cfg.py` | `True` | `Ceiling`, `Wall`, `Kitchen/Kitchen_Cabinet002`, `Kitchen/Stovetop017`, `Kitchen/SeasoningBox001`, `Kitchen/Jar046`, `Kitchen/Shovel008`, `Kitchen/Shovel007`, `Kitchen/WoodenSpoon013` |
| `source/lehome/lehome/tasks/kitchen/loft_cut_bi_cfg.py` | `True` | `Ceiling`, `Wall`, `Kitchen/Kitchen_Cabinet002`, `Kitchen/Stovetop017`, `Kitchen/SeasoningBox001`, `Kitchen/Jar046`, `Kitchen/Shovel008`, `Kitchen/Shovel007`, `Kitchen/WoodenSpoon013` |
| `source/lehome/lehome/tasks/kitchen/loft_fire_bi_cfg.py` | `True` | `Ceiling`, `Wall`, `Kitchen/Kitchen_Cabinet002`, `Kitchen/Stovetop017`, `Kitchen/ToasterOven009` |
| `source/lehome/lehome/tasks/livingroom/loft_water_cfg.py` | `True` | `Ceiling`, `Wall`, `Livingroom/Table062`, `Chair024`, `Chair024_01`, `Chair024_02`, `Chair024_03` |
| `source/lehome/lehome/tasks/washroom/loft_wipe_cfg.py` | `True` | `Ceiling`, `Wall`, `Washroom/Toilet033` |

The names above are abbreviated for readability. The code uses full paths under `/World/Scene/...`.

## 6. Example: Wiping Task

The wiping task keeps only the washroom subtree that is required by the task.

Example configuration:

```python
scene_deactivation_enabled: bool = True
scene_deactivation_root_path: str = "/World/Scene"
scene_keep_prim_path_prefixes: tuple[str, ...] = (
    "/World/Scene/Ceiling",
    "/World/Scene/Wall",
    "/World/Scene/Washroom/Toilet033",
)
scene_deactivation_log_limit: int = 8
```

This configuration keeps:

- the ceiling
- the wall
- the toilet subtree in the washroom

Everything else under `/World/Scene` is deactivated unless it is needed as a bridge node.

## 7. Example: Keep an Entire Room

If you want to keep the whole washroom active, whitelist the room root instead of a single object:

```python
scene_keep_prim_path_prefixes: tuple[str, ...] = (
    "/World/Scene/Ceiling",
    "/World/Scene/Wall",
    "/World/Scene/Washroom",
)
```

In this case:

- the whole washroom remains visible and simulated
- unrelated rooms can still be deactivated

## 8. Startup Logs

When enabled, startup prints summary logs such as:

```text
[LoftWipeEnv][SceneDeactivate] processed=..., deactivated=..., kept=..., bridges=..., already_inactive=...
```

Depending on `scene_deactivation_log_limit`, the base environment may also print:

- a sample of deactivated paths
- a sample of kept paths
- a sample of bridge paths
- a sample of already inactive paths

These logs are useful when tuning whitelist prefixes for a new task.
