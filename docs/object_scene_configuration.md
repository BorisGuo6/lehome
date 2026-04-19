# Object and Scene Configuration Guide

This guide explains the recommended ways to add task objects and scene assets in LeHome.

When a task reuses assets that already exist under `/World/Scene`, this guide only covers the object-configuration dependency at a high level. For the full deactivation mechanism, traversal rules, and logging behavior, see [Scene Deactivation Guide](scene_deactivation.md).

The current repository uses three main patterns:

1. `DeformableObjectCfg` / `RigidObjectCfg` for most deformable and rigid task objects.
2. `GarmentObject` / `FluidObject` for cloth-like and fluid-like particle-based objects implemented by LeHome.
3. Reuse existing shared-scene assets such as `Stovetop017` and `ToasterOven009`, and add keep-path configuration when scene deactivation is enabled.

## Table of Contents

- [Object and Scene Configuration Guide](#object-and-scene-configuration-guide)
  - [Table of Contents](#table-of-contents)
  - [1. Overview](#1-overview)
  - [2. Core Distinction: Task Objects vs. Scene Assets](#2-core-distinction-task-objects-vs-scene-assets)
  - [3. Method 1: Use `DeformableObjectCfg` and `RigidObjectCfg`](#3-method-1-use-deformableobjectcfg-and-rigidobjectcfg)
    - [3.1 When to Use](#31-when-to-use)
    - [3.2 Configuration Pattern](#32-configuration-pattern)
    - [3.3 Example: loft_burger_bi](#33-example-loft_burger_bi)
    - [3.4 Practical Notes](#34-practical-notes)
  - [4. Method 2: Use `GarmentObject` and `FluidObject`](#4-method-2-use-garmentobject-and-fluidobject)
    - [4.1 When to Use](#41-when-to-use)
    - [4.2 Configuration Pattern](#42-configuration-pattern)
    - [4.3 Example: loft_wipe](#43-example-loft_wipe)
    - [4.4 Practical Notes](#44-practical-notes)
  - [5. Method 3: Reuse `Existing Shared-Scene Assets`](#5-method-3-reuse-existing-shared-scene-assets)
    - [5.1 When to Use](#51-when-to-use)
    - [5.2 Required Steps](#52-required-steps)
    - [5.3 Example: loft_fire_bi](#53-example-loft_fire_bi)
    - [5.4 Practical Notes](#54-practical-notes)
  - [6. Recommended Decision Flow](#6-recommended-decision-flow)
  - [7. Asset Library Categories and Example Assets](#7-asset-library-categories-and-example-assets)
    - [7.1 Asset Library Overview](#71-asset-library-overview)
    - [7.2 Fluids](#72-fluids)
    - [7.3 Granular](#73-granular)
    - [7.4 Linear_Objects](#74-linear_objects)
    - [7.5 Plasmas](#75-plasmas)
    - [7.6 Thin-Shells](#76-thin-shells)
    - [7.7 Volumetric_Objects](#77-volumetric_objects)
    - [7.8 Diverse_Manipulation_Mechanisms](#78-diverse_manipulation_mechanisms)
  - [8. Best Practices](#8-best-practices)

## 1. Overview

LeHome loads the shared apartment scene in `BaseEnv._setup_scene()` from:

```text
Assets/scenes/1BRAPT_LeHome/1BRAPT_LeHome.usd
```

This scene is attached under:

```text
/World/Scene
```

After the shared scene is loaded, each task adds its own robots, sensors, manipulated objects, and any task-specific logic inside the task environment's `_setup_scene()` implementation.

When configuring a new task, the first design decision is:

- whether the asset should be spawned as a task-local object
- or whether the asset already belongs to the shared scene and only needs to be kept active for the task

That distinction determines which configuration pattern should be used.

## 2. Core Distinction: Task Objects vs. Scene Assets

In LeHome, it is important to separate the following two cases:

- **Task objects**: assets that are added specifically for a task, usually under prim paths such as `/World/Burger`, `/World/Cut`, or `/World/Object`
- **Scene assets**: assets that already exist inside the shared apartment scene under `/World/Scene/...`

This distinction matters because LeHome can deactivate irrelevant subtrees under `/World/Scene`.

As a result:

- objects spawned outside `/World/Scene` are not part of shared-scene keep-path configuration
- assets that live inside `/World/Scene` may need to be added to `scene_keep_prim_path_prefixes` when scene deactivation is enabled

For the detailed behavior of scene deactivation, see [Scene Deactivation Guide](scene_deactivation.md).

Therefore, before writing code, first confirm whether the target asset should be:

1. spawned as a new task object, or
2. reused from the existing shared scene

## 3. Method 1: Use `DeformableObjectCfg` and `RigidObjectCfg`

### 3.1 When to Use

This is the recommended pattern for most ordinary deformable objects and rigid objects in LeHome.

Use this method when:

- the asset should be spawned specifically for the task
- the asset can be represented directly by Isaac Lab's standard object configuration classes
- the object does not require LeHome's custom cloth or fluid particle wrappers

In practice, this is the default choice for most new deformable or rigid task objects.

### 3.2 Configuration Pattern

The usual pattern has two steps:

1. Define the object configuration in `<Task>EnvCfg`.
2. Instantiate the object in `<Task>Env._setup_scene()`.

The configuration normally contains:

- `prim_path`: where the object will be created in the stage
- `spawn=sim_utils.UsdFileCfg(...)`: which USD file to load
- `init_state`: initial position and orientation

### 3.3 Example: `loft_burger_bi`

In `LoftBurgerEnvCfg`, the task defines the object configurations first:

```python
burger_beef: DeformableObjectCfg = DeformableObjectCfg(
    prim_path="/World/Burger/burger_beef",
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.getcwd()
        + "/Assets/objects/Volumetric_Objects/burger/Assets/Burger_Beef_Patties001/Burger_Beef_Patties001_Def.usd"
    ),
    init_state=DeformableObjectCfg.InitialStateCfg(
        pos=(-3.76, 6.8, 0.828),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)

burger_board: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/Burger/burger_board",
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.getcwd()
        + "/Assets/objects/Volumetric_Objects/burger/Assets/Burger_ChoppingBlock/Burger_ChoppingBlock.usd"
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(-3.75, 6.7, 0.81),
        rot=(0, 0.0, 0.0, 1),
    ),
)
```

Then, in `LoftBurgerEnv._setup_scene()`, the task instantiates them:

```python
self.burger_beef = DeformableObject(self.cfg.burger_beef)
self.burger_board = RigidObject(self.cfg.burger_board)
```

### 3.4 Practical Notes

- This method is preferred for most task-local deformable and rigid assets because it follows Isaac Lab's standard asset lifecycle.
- The object is configured declaratively in the env config, which makes the task definition easier to read and maintain.
- In this example, the objects are created under `/World/Burger`, so they are outside `/World/Scene` and do not require shared-scene keep-path configuration.
- `prim_path` should be unique inside the stage.
- `init_state` should match the object type:
  - use `DeformableObjectCfg.InitialStateCfg` for deformable assets
  - use `RigidObjectCfg.InitialStateCfg` for rigid assets

## 4. Method 2: Use `GarmentObject` and `FluidObject`

### 4.1 When to Use

This method is recommended for objects whose physical behavior is implemented through LeHome's custom particle-based wrappers, especially:

- cloth or garment-like assets
- fluid assets

Use this method when the object requires:

- custom particle system setup
- task-specific YAML configuration loaded through `OmegaConf`
- custom initialization and reset behavior provided by LeHome classes

### 4.2 Configuration Pattern

Unlike Method 1, these objects are not typically declared through `DeformableObjectCfg` or `RigidObjectCfg`.

Instead, the task directly constructs them inside `_setup_scene()` by passing:

- `prim_path`
- `usd_path`
- optional material or visual USD paths
- YAML config loaded by `OmegaConf.load(...)`
- any extra constructor arguments required by the custom class

### 4.3 Example: `loft_wipe`

In `LoftWipeEnv._setup_scene()`, the towel and water are instantiated directly:

```python
self.towel = GarmentObject(
    prim_path="/World/Objects/Towel",
    usd_path=os.getcwd() + "/Assets/objects/Thin-Shells/Towel/towel.usd",
    visual_usd_path=os.getcwd() + "/Assets/Material/Garment/linen_Blue.usd",
    config=OmegaConf.load(
        os.getcwd()
        + "/source/lehome/lehome/tasks/washroom/config_file/particle_towel_cfg.yaml"
    ),
)

self.object = FluidObject(
    env_id=0,
    env_origin=torch.zeros(1, 3),
    prim_path="/World/Object/fluid_items/fluid_items_1",
    usd_path=os.getcwd() + "/Assets/objects/Fluids/water/water.usdc",
    config=OmegaConf.load(
        os.getcwd()
        + "/source/lehome/lehome/tasks/washroom/config_file/fluid.yaml"
    ),
    use_container=False,
)
```

The same task also calls the corresponding lifecycle methods later:

```python
def _reset_idx(self, env_ids: Sequence[int] | None):
    ...
    self.towel.reset()
    self.object.reset(soft=True)

def initialize_obs(self):
    self.object.initialize()
    self.towel.initialize()
```

### 4.4 Practical Notes

- Use this method only when the object truly depends on LeHome's custom garment or fluid implementation.
- `GarmentObject` is appropriate for cloth-like particle cloth assets.
- `FluidObject` is appropriate for particle-based fluid assets and supports additional behavior such as optional containers.
- The YAML file is part of the object definition, not just an optional helper. It controls material, particle, and object behavior parameters.
- In the current tasks, these objects are placed under `/World/Objects` or `/World/Object`, so they do not require shared-scene keep-path configuration.

## 5. Method 3: Reuse Existing Shared-Scene Assets

### 5.1 When to Use

This method is recommended for fixed assets that already belong to the shared scene and should not be spawned as a separate copy.

Typical examples in LeHome include:

- `Stovetop017`
- `ToasterOven009`

Use this method when:

- the asset already exists in `1BRAPT_LeHome.usd` or in the scene assets referenced by it
- the asset lives under `/World/Scene/...`
- the task needs that asset to remain available during execution
- the asset may contain authored joints, location, or action-graph-related functionality that should be reused from the existing scene asset

### 5.2 Required Steps

For this kind of asset, the recommended workflow is:

1. Confirm that the asset is present in the shared scene hierarchy.
2. If it is not present, import the needed asset into `Assets/scenes/1BRAPT_LeHome/1BRAPT_LeHome.usd` first. In the current repository layout, assets under `Assets/scenes/1BRAPT_LeHome/Assets` are already preloaded by the base scene, while additional optional assets are available under `Assets/scenes/1BRAPT_LeHome/Assets_not_loaded`.
3. If `scene_deactivation_enabled=True`, add the asset path to `scene_keep_prim_path_prefixes`.
4. In `_setup_scene()`, complete the task-side scene configuration using that kept scene asset.

The key point is that this method reuses an existing shared-scene prim. It does not spawn a new task-local copy with `UsdFileCfg`.

For the exact deactivation behavior and keep-path rules, see [Scene Deactivation Guide](scene_deactivation.md).

### 5.3 Example: `loft_fire_bi`

In `LoftFireEnvCfg`, the stovetop path is added to the keep-path configuration:

```python
scene_keep_prim_path_prefixes: tuple[str, ...] = (
    "/World/Scene/Kitchen/Stovetop017",
)
```

In practice, a task may keep additional shared-scene prefixes depending on the room layout and task dependencies. For `loft_fire_bi`, the repository keeps several related scene paths, including:

```python
scene_keep_prim_path_prefixes: tuple[str, ...] = (
    "/World/Scene/Ceiling",
    "/World/Scene/Wall",
    "/World/Scene/Kitchen/Kitchen_Cabinet002",
    "/World/Scene/Kitchen/Stovetop017",
)
```

Then, in `LoftFireEnv._setup_scene()`, the task continues its scene setup based on this kept stovetop asset.

### 5.4 Practical Notes

- This method is relevant for assets that already live inside `/World/Scene`.
- If scene deactivation is enabled and the required path is missing from `scene_keep_prim_path_prefixes`, the asset may become unavailable before the task uses it.
- For fixed scene assets, prefer reusing the existing scene prim rather than spawning another copy elsewhere unless there is a strong reason to duplicate it.
- Keep-path configuration is the critical integration point for shared-scene assets when scene deactivation is enabled.
- After the asset is kept active, the task can finish its own scene setup around that existing scene asset.
- Refer to [Scene Deactivation Guide](scene_deactivation.md) for deactivation scope, bridge-node behavior, and startup logs.

## 6. Recommended Decision Flow

Use the following rule of thumb when adding a new object or scene asset:

1. If the asset already exists under `/World/Scene` and should be reused from the base apartment scene, use **Method 3**.
2. If the asset is a cloth-like or fluid-like object that depends on LeHome's custom particle wrappers, use **Method 2**.
3. For most other deformable or rigid task-local objects, use **Method 1**.

This is the recommended priority order for the current LeHome repository.

## 7. Asset Library Categories and Example Assets

### 7.1 Asset Library Overview

Under `Assets/objects`, LeHome organizes reusable assets into seven categories:

- `Diverse_Manipulation_Mechanisms`
- `Fluids`
- `Granular`
- `Linear_Objects`
- `Plasmas`
- `Thin-Shells`
- `Volumetric_Objects`

Simulated Deformble Objects of LeHome cover 6 categories, namely `Fluids`, `Granular`, `Linear_Objects`, `Plasmas`, `Thin-Shells`, and `Volumetric_Objects`, with a large number of visually and physically high-fidelity assets for each category.

`Diverse_Manipulation_Mechanisms`. LeHome models causal relationships of manipulation through the action graph mechanism, ensuring the simulation results align with real-world causal relationships and providing high-fidelity interactions.

The categorized assets under `Assets/objects` are curated from the broader shared scene-asset libraries in:

```text
Assets/scenes/1BRAPT_LeHome/Assets
Assets/scenes/1BRAPT_LeHome/Assets_not_loaded
```

This means:

- `Assets/objects` provides task-oriented example assets that are easier to discover and reuse
- `Assets/scenes/1BRAPT_LeHome/Assets` contains assets already used by the base apartment scene
- `Assets/scenes/1BRAPT_LeHome/Assets_not_loaded` contains additional assets that can be imported into `1BRAPT_LeHome.usd` when a task needs them

The tables below list all currently available top-level asset folders for each category. For consistency, the `Object name` column uses the first-level folder name directly. These recommendations follow the three methods described earlier in this guide:

- **Method 1**: `DeformableObjectCfg` / `RigidObjectCfg`
- **Method 2**: `GarmentObject` / `FluidObject`
- **Method 3**: reuse the shared-scene asset and configure keep-paths when needed

Unless an asset already has a dedicated LeHome wrapper or is intended to be reused directly from `/World/Scene`, the default recommendation is Method 1.

### 7.2 `Fluids`

| Object name | Recommended method | Path |
|-------------|--------------------|------|
| `water` | Method 2 | `Assets/objects/Fluids/water` |
| `Bowl016` | Method 1 | `Assets/objects/Fluids/Bowl016` |
| `Cup012` | Method 1 | `Assets/objects/Fluids/Cup012` |
| `GlassCup007` | Method 1 | `Assets/objects/Fluids/GlassCup007` |
| `GourdLadle001` | Method 1 | `Assets/objects/Fluids/GourdLadle001` |
| `Teapot029` | Method 1 | `Assets/objects/Fluids/Teapot029` |

### 7.3 `Granular`

| Object name | Recommended method | Path |
|-------------|--------------------|------|
| `CoffeeBeans001` | Method 1 | `Assets/objects/Granular/CoffeeBeans001` |
| `MungBean001` | Method 1 | `Assets/objects/Granular/MungBean001` |
| `RedBean001` | Method 1 | `Assets/objects/Granular/RedBean001` |
| `Soybean001` | Method 1 | `Assets/objects/Granular/Soybean001` |
| `towel_particle_erasing_01(with dust)` | Method 3 | `Assets/objects/Granular/DishTowel003` |


### 7.4 `Linear_Objects`

| Object name | Recommended method | Path |
|-------------|--------------------|------|
| `Cable` | Method 1 | `Assets/objects/Linear_Objects/Cable` |
| `PaperBag（with rope）` | Method 3 | `Assets/objects/Linear_Objects/PaperBag` |
| `Television008（with cable）` | Method 3 | `Assets/objects/Linear_Objects/Television008` |
| `Blinds001（with rope）` | Method 3 | `Assets/objects/Linear_Objects/Blinds001` |

### 7.5 `Plasmas`

| Object name | Recommended method | Path |
|-------------|--------------------|------|
| `Stovetop017（with fire）` | Method 3 | `Assets/objects/Plasmas/Stovetop017` |
| `Fire` | Method 3 | `Assets/objects/Plasmas/Fire` |

### 7.6 `Thin-Shells`

| Object name | Recommended method | Path |
|-------------|--------------------|------|
| `Towel` | Method 2 | `Assets/objects/Thin-Shells/Towel` |
| `garment` | Method 2 | `Assets/objects/Thin-Shells/garment` |
| `BedTowel001` | Method 2 | `Assets/objects/Thin-Shells/BedTowel001` |
| `DishTowel003` | Method 2 | `Assets/objects/Thin-Shells/DishTowel003` |
| `DishTowel004` | Method 2 | `Assets/objects/Thin-Shells/DishTowel004` |
| `Package001` | Method 2 | `Assets/objects/Thin-Shells/Package001` |
| `Paper` | Method 2 | `Assets/objects/Thin-Shells/Paper` |
| `Towel008` | Method 2 | `Assets/objects/Thin-Shells/Towel008` |
| `Towel009` | Method 2 | `Assets/objects/Thin-Shells/Towel009` |

### 7.7 `Volumetric_Objects`

| Object name | Recommended method | Path |
|-------------|--------------------|------|
| `DesktopBroom001` | Method 1 | `Assets/objects/Volumetric_Objects/DesktopBroom001` |
| `BaggedFood019` | Method 1 | `Assets/objects/Volumetric_Objects/BaggedFood019` |
| `BaggedFood020` | Method 1 | `Assets/objects/Volumetric_Objects/BaggedFood020` |
| `BeefPatties001` | Method 1 | `Assets/objects/Volumetric_Objects/BeefPatties001` |
| `Candy012` | Method 1 | `Assets/objects/Volumetric_Objects/Candy012` |
| `Pillow002` | Method 1 | `Assets/objects/Volumetric_Objects/Pillow002` |
| `Pillow003` | Method 1 | `Assets/objects/Volumetric_Objects/Pillow003` |
| `Pillow004` | Method 1 | `Assets/objects/Volumetric_Objects/Pillow004` |
| `Pillow005` | Method 1 | `Assets/objects/Volumetric_Objects/Pillow005` |
| `Doll002` | Method 1 | `Assets/objects/Volumetric_Objects/Doll002` |
| `Doll003` | Method 1 | `Assets/objects/Volumetric_Objects/Doll003` |
| `Doll004` | Method 1 | `Assets/objects/Volumetric_Objects/Doll004` |
| `NewsPaper002` | Method 1 | `Assets/objects/Volumetric_Objects/NewsPaper002` |
| `Quilt002` | Method 1 | `Assets/objects/Volumetric_Objects/Quilt002` |
| `SandwichBread014` | Method 1 | `Assets/objects/Volumetric_Objects/SandwichBread014` |
| `Sausage001` | Method 1 | `Assets/objects/Volumetric_Objects/Sausage001` |
| `Sponge001` | Method 1 | `Assets/objects/Volumetric_Objects/Sponge001` |
| `Toothpaste001` | Method 1 | `Assets/objects/Volumetric_Objects/Toothpaste001` |
| `burger` | Method 1 | `Assets/objects/Volumetric_Objects/burger` |

### 7.8 `Diverse_Manipulation_Mechanisms`

| Object name | Recommended method | Path |
|-------------|--------------------|------|
| `Blinds001` | Method 3 | `Assets/objects/Diverse_Manipulation_Mechanisms/Blinds001` |
| `Microwave059` | Method 3 | `Assets/objects/Diverse_Manipulation_Mechanisms/Microwave059` |
| `Refrigerator066` | Method 3 | `Assets/objects/Diverse_Manipulation_Mechanisms/Refrigerator066` |
| `Sink054` | Method 3 | `Assets/objects/Diverse_Manipulation_Mechanisms/Sink054` |
| `Stovetop017` | Method 3 | `Assets/objects/Diverse_Manipulation_Mechanisms/Stovetop017` |
| `ToasterOven009` | Method 3 | `Assets/objects/Diverse_Manipulation_Mechanisms/ToasterOven009` |
| `Toaster_Scene` | Method 3 | `Assets/objects/Diverse_Manipulation_Mechanisms/Toaster_Scene` |
| `WashingMachine029` | Method 3 | `Assets/objects/Diverse_Manipulation_Mechanisms/WashingMachine029` |

## 8. Best Practices

- Decide early whether an asset is a task-local object or a shared scene asset. This avoids duplicated imports and conflicting prim paths.
- Keep task-local objects outside `/World/Scene` unless they are intentionally part of the shared scene hierarchy.
- When a task reuses shared-scene assets, always verify whether `scene_keep_prim_path_prefixes` must include them.
- Use unique and descriptive `prim_path` values so later reset and debugging logic remains clear.
- Keep asset declaration and object instantiation consistent:
  - Method 1: declare in `<Task>EnvCfg`, instantiate in `_setup_scene()`
  - Method 2: construct directly in `_setup_scene()` and manage custom lifecycle methods
  - Method 3: reuse the shared-scene prim, configure keep-paths when needed, then complete the task-side setup in `_setup_scene()`
- Validate the final setup through startup logs, visibility checks, reset behavior, and task interaction behavior.
