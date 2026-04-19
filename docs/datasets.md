# Dataset Collection and Processing Guide

This guide covers how to collect teleoperation demonstration data and process datasets for training.

## Table of Contents

- [Dataset Collection and Processing Guide](#dataset-collection-and-processing-guide)
  - [Table of Contents](#table-of-contents)
  - [1. Overview](#1-overview)
  - [2. Hardware Setup](#2-hardware-setup)
    - [2.1 Hardware Connection](#21-hardware-connection)
    - [2.2 Calibration](#22-calibration)
    - [2.3 Verify Calibration](#23-verify-calibration)
  - [3. Teleoperation Data Collection](#3-teleoperation-data-collection)
    - [3.1 Basic Recording Command](#31-basic-recording-command)
    - [3.2 Parameter Configuration](#32-parameter-configuration)
    - [3.3 Recording Controls](#33-recording-controls)
    - [3.4 Task Examples](#34-task-examples)
  - [4. Dataset Inspection and Processing](#4-dataset-inspection-and-processing)
    - [4.1 Inspect Dataset](#41-inspect-dataset)
    - [4.2 Read Dataset States](#42-read-dataset-states)
    - [4.3 Add End-Effector Pose](#43-add-end-effector-pose)
    - [4.4 Add PointCloud](#44-add-pointcloud)
    - [4.5 Replay Dataset](#45-replay-dataset)
    - [4.6 Remove Features](#46-remove-features)
  - [5. Dataset Merging](#5-dataset-merging)
  - [6. Dataset Format](#6-dataset-format)
    - [6.1 Data Features](#61-data-features)
    - [6.2 Metadata](#62-metadata)
  - [7. Best Practices](#7-best-practices)

## 1. Overview

LeHome supports collecting demonstration data through teleoperation using SO101 Leader Arms. The collected data is saved in LeRobot dataset format and can be processed and merged for training.

The current task IDs registered in this repository are:

| Task ID | Arms | Recommended teleop device | Recommended simulator device in this repo |
|---------|------|----------------------------|-------------------------------------------|
| `LeHome-BiSO101-Direct-Garment-v0` | Dual-arm | `bi-so101leader` | `cpu` |
| `LeHome-BiSO101-Direct-Garment-fling-v0` | Dual-arm | `bi-so101leader` | `cpu` |
| `LeHome-BiSO101-Direct-loftcut-v0` | Dual-arm | `bi-so101leader` | `cpu` |
| `LeHome-BiSO101-Direct-loftfire-v0` | Dual-arm | `bi-so101leader` | `cpu` |
| `LeHome-BiSO101-Direct-loftburger-v0` | Dual-arm | `bi-so101leader` | `cuda` |
| `LeHome-SO101-Direct-loftwater-v0` | Single-arm | `so101leader` | `cpu` |
| `LeHome-SO101-Direct-loftwipe-v0` | Single-arm | `so101leader` | `cpu` |

## 2. Hardware Setup

This section covers the setup and configuration of SO101 Leader Arms for teleoperation.

### 2.1 Hardware Connection

For **SO101 Leader Arms** teleoperation:

1. Connect the leader arm or leader-arm pair to the computer via USB.
2. Ensure the devices are powered correctly.
3. Identify the serial ports:

```bash
ls /dev/ttyACM*
# Typical mapping:
#   /dev/ttyACM0 for a single arm
#   /dev/ttyACM0 and /dev/ttyACM1 for dual-arm setup
```

4. Grant serial permissions if needed:

```bash
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
```

Recommended hardware list:

- 1 or 2 SO101 leader arms
- the corresponding SO101 follower embodiment in simulation
- USB-to-serial connection for each leader arm
- external power as required by your hardware setup

### 2.2 Calibration

**First-time use** or after **changing hardware**, calibrate the SO101 Leader Arms:

Dual-arm calibration:

```bash
python -m scripts.dataset_sim record \
    --task LeHome-BiSO101-Direct-Garment-v0 \
    --teleop_device bi-so101leader \
    --left_arm_port /dev/ttyACM0 \
    --right_arm_port /dev/ttyACM1 \
    --device cpu \
    --enable_cameras \
    --recalibrate
```

Single-arm calibration:

```bash
python -m scripts.dataset_sim record \
    --task LeHome-SO101-Direct-loftwipe-v0 \
    --teleop_device so101leader \
    --port /dev/ttyACM0 \
    --device cpu \
    --enable_cameras \
    --recalibrate
```

Calibration files are stored under:

```text
source/lehome/lehome/devices/lerobot/.cache/
```

Verified filenames in the current code:

- single-arm leader: `so101_leader.json`
- dual-arm left leader: `left_so101_leader.json`
- dual-arm right leader: `right_so101_leader.json`

**Note:** Calibration is required only once. Recalibrate if hardware is replaced or control feels inaccurate.

### 2.3 Verify Calibration

Test the hardware calibration without recording:

Dual-arm:

```bash
python -m scripts.dataset_sim record \
    --task LeHome-BiSO101-Direct-Garment-v0 \
    --teleop_device bi-so101leader \
    --left_arm_port /dev/ttyACM0 \
    --right_arm_port /dev/ttyACM1 \
    --device cpu \
    --enable_cameras
```

Single-arm:

```bash
python -m scripts.dataset_sim record \
    --task LeHome-SO101-Direct-loftwater-v0 \
    --teleop_device so101leader \
    --port /dev/ttyACM0 \
    --device cpu \
    --enable_cameras
```

Verification checklist:

1. Isaac Sim opens correctly.
2. Moving the physical leader arm updates the simulated robot.
3. Left/right mapping is correct for the dual-arm setup.
4. Press `Ctrl+C` to exit when finished.

## 3. Teleoperation Data Collection

### 3.1 Basic Recording Command

**Dual-arm recording template:**

```bash
python -m scripts.dataset_sim record \
    --task <DUAL_ARM_TASK_ID> \
    --teleop_device bi-so101leader \
    --left_arm_port /dev/ttyACM0 \
    --right_arm_port /dev/ttyACM1 \
    --enable_record \
    --num_episode <NUM_EPISODES> \
    --dataset_root <OUTPUT_PARENT_DIR> \
    --task_description "<TASK_DESCRIPTION>" \
    --device <sim_device> \
    --enable_cameras
```

**Single-arm recording template:**

```bash
python -m scripts.dataset_sim record \
    --task <SINGLE_ARM_TASK_ID> \
    --teleop_device so101leader \
    --port /dev/ttyACM0 \
    --enable_record \
    --num_episode <NUM_EPISODES> \
    --dataset_root <OUTPUT_PARENT_DIR> \
    --task_description "<TASK_DESCRIPTION>" \
    --device <sim_device> \
    --enable_cameras
```

The recorder automatically creates the next available numbered subdirectory under `--dataset_root`, such as `Datasets/record/fold_garment/001`.
Use `cpu`, `cuda`, or `cuda:N` for `<sim_device>` according to the task and runtime environment.

### 3.2 Parameter Configuration

**Input Device Options:**

| Parameter | Single-Arm | Dual-Arm | Description |
|-----------|------------|----------|-------------|
| `--teleop_device` | `so101leader` | `bi-so101leader` | Recommended SO101 leader teleoperation mode |
| `--teleop_device` | `keyboard` | `bi-keyboard` | Keyboard teleoperation remains implemented in code, but is not the default in this documentation |
| `--port` | `/dev/ttyACM0` | - | Single-arm leader serial port |
| `--left_arm_port` | - | `/dev/ttyACM0` | Dual-arm left leader port |
| `--right_arm_port` | - | `/dev/ttyACM1` | Dual-arm right leader port |
| `--recalibrate` | Optional | Optional | Rebuild calibration files |

**Task Configuration:**

| Parameter | Verified values or default | Description |
|-----------|----------------------------|-------------|
| `--task` | `LeHome-BiSO101-Direct-Garment-v0`<br>`LeHome-BiSO101-Direct-Garment-fling-v0`<br>`LeHome-BiSO101-Direct-loftcut-v0`<br>`LeHome-BiSO101-Direct-loftfire-v0`<br>`LeHome-BiSO101-Direct-loftburger-v0`<br>`LeHome-SO101-Direct-loftwater-v0`<br>`LeHome-SO101-Direct-loftwipe-v0` | Task ID registered |
| `--task_description` | default empty string | Task label for filtering |

**Recording Options:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--enable_record` | `disabled` | Enable dataset recording |
| `--num_episode` | `20` | Number of successful episodes to save |
| `--dataset_root` | `Datasets/record` | Parent directory for numbered dataset outputs |
| `--disable_depth` | `False` | Do not save `observation.top_depth` |
| `--enable_pointcloud` | `False` | Please convert depth to pointcloud offline following [Add pointcloud](#44-add-pointcloud) |
| `--record_ee_pose` | `False` | Add `observation.ee_pose` and `action.ee_pose` during recording |
| `--ee_urdf_path` | `Assets/robots/so101_new_calib.urdf` | URDF used when `--record_ee_pose` is enabled |
| `--ee_state_unit` | `rad` | Joint unit for kinematics, choices: `rad`, `deg` |

**Simulation Options (Isaac Lab AppLauncher):**

| Parameter | Verified values or default | Description |
|-----------|----------------------------|-------------|
| `--enable_cameras` | default `False` | Enable camera sensors and camera rendering |
| `--headless` | default `False` | Run without GUI |
| `--device` | default `cuda:0` | Simulator device, choices: `cpu`, `cuda` |

### 3.3 Recording Controls

**Function Keys:**

- **B key**: Start teleoperation (activate control, must be pressed before `S`)
- **S key**: Start recording current episode
- **N key**: Save current episode (mark as success)
- **D key**: Discard current episode (re-record)
- **ESC key**: Abort recording and clear buffer
- **Ctrl+C**: Exit program

**Keyboard Control (for keyboard/bi-keyboard):**

- `keyboard` (single-arm):
    - Joint 1 (shoulder_pan): `T / G`
    - Joint 2 (shoulder_lift): `Y / H`
    - Joint 3 (elbow_flex): `U / J`
    - Joint 4 (wrist_flex): `I / K`
    - Joint 5 (wrist_roll): `O / L`
    - Joint 6 (gripper): `Q / A`
- `bi-keyboard` (dual-arm):
    - Left arm: same mapping as single-arm (`T/G`, `Y/H`, `U/J`, `I/K`, `O/L`, `Q/A`)
    - Right arm:
        - Joint 1: `1 / 2` (also supports numpad `1 / 2`)
        - Joint 2: `3 / 4` (also supports numpad `3 / 4`)
        - Joint 3: `5 / 6` (also supports numpad `5 / 6`)
        - Joint 4: `7 / 8` (also supports numpad `7 / 8`)
        - Joint 5: `9 / 0` (also supports numpad `9 / 0`)
        - Joint 6 (gripper): `[` / `]` (also supports numpad `+ / -`)

**SO101 Leader Control:**
- Directly move the physical Leader Arms; the simulated robots follow in real-time

### 3.4 Task Examples

Dual-arm garment fold:

```bash
python -m scripts.dataset_sim record \
    --task LeHome-BiSO101-Direct-Garment-v0 \
    --teleop_device bi-so101leader \
    --left_arm_port /dev/ttyACM0 \
    --right_arm_port /dev/ttyACM1 \
    --enable_record \
    --num_episode 10 \
    --dataset_root Datasets/record/fold_garment \
    --task_description "fold the garment on the table" \
    --device cpu \
    --enable_cameras
```

Dual-arm garment fling:

```bash
python -m scripts.dataset_sim record \
    --task LeHome-BiSO101-Direct-Garment-fling-v0 \
    --teleop_device bi-so101leader \
    --left_arm_port /dev/ttyACM0 \
    --right_arm_port /dev/ttyACM1 \
    --enable_record \
    --num_episode 10 \
    --dataset_root Datasets/record/fling_garment \
    --task_description "fling the garment on the table" \
    --device cpu \
    --enable_cameras
```

Dual-arm sausage cutting:

```bash
python -m scripts.dataset_sim record \
    --task LeHome-BiSO101-Direct-loftcut-v0 \
    --teleop_device bi-so101leader \
    --left_arm_port /dev/ttyACM0 \
    --right_arm_port /dev/ttyACM1 \
    --enable_record \
    --num_episode 10 \
    --dataset_root Datasets/record/loftcut \
    --task_description "cut the sausage on the chopping board" \
    --device cpu \
    --enable_cameras
```

Dual-arm stove ignition:

```bash
python -m scripts.dataset_sim record \
    --task LeHome-BiSO101-Direct-loftfire-v0 \
    --teleop_device bi-so101leader \
    --left_arm_port /dev/ttyACM0 \
    --right_arm_port /dev/ttyACM1 \
    --enable_record \
    --num_episode 10 \
    --dataset_root Datasets/record/fire \
    --task_description "turn the knob to light the gas stove" \
    --device cpu \
    --enable_cameras
```

Dual-arm burger assembly:

```bash
python -m scripts.dataset_sim record \
    --task LeHome-BiSO101-Direct-loftburger-v0 \
    --teleop_device bi-so101leader \
    --left_arm_port /dev/ttyACM0 \
    --right_arm_port /dev/ttyACM1 \
    --enable_record \
    --num_episode 10 \
    --dataset_root Datasets/record/burger \
    --task_description "place the patty on the burger" \
    --device cuda \
    --enable_cameras
```

Single-arm water pouring:

```bash
python -m scripts.dataset_sim record \
    --task LeHome-SO101-Direct-loftwater-v0 \
    --teleop_device so101leader \
    --port /dev/ttyACM0 \
    --enable_record \
    --num_episode 10 \
    --dataset_root Datasets/record/water \
    --task_description "pour water from cup into bowl" \
    --device cpu \
    --enable_cameras
```

Single-arm wiping:

```bash
python -m scripts.dataset_sim record \
    --task LeHome-SO101-Direct-loftwipe-v0 \
    --teleop_device so101leader \
    --port /dev/ttyACM0 \
    --enable_record \
    --num_episode 10 \
    --dataset_root Datasets/record/wipe \
    --task_description "wipe the water stains clean" \
    --device cpu \
    --enable_cameras
```

## 4. Dataset Inspection and Processing

The following commands use `Datasets/record/fold_garment/001` as an example dataset.

### 4.1 Inspect Dataset

View dataset metadata, frame data, and statistics:

```bash
python -m scripts.dataset inspect \
    --dataset_root Datasets/record/fold_garment/001 \
    --show_frames 3 \
    --show_stats
```

**Additional Parameters:**

- `--show_frames N`: Display the first N frames of sample data
- `--show_stats`: Display detailed statistical information for numeric columns

### 4.2 Read Dataset States

Read and analyze dataset observation/action data:

```bash
python -m scripts.dataset read \
    --dataset_root Datasets/record/fold_garment/001 \
    --num_frames 10 \
    --episode 0 \
    --show_stats
```

**Additional Parameters:**

- `--num_frames N`: Number of frames to display
- `--episode N`: Read a specific episode index
- `--show_stats`: Display statistical information

### 4.3 Add End-Effector Pose

Add end-effector pose to an existing dataset offline (computes both `observation.ee_pose` and `action.ee_pose`):

```bash
python -m scripts.dataset augment \
    --dataset_root Datasets/record/fold_garment/001 \
    --urdf_path Assets/robots/so101_new_calib.urdf \
    --state_unit rad \
    --overwrite
```

**Parameters:**

- `--urdf_path`: Robot URDF file path
- `--state_unit`: Joint angle unit (`rad` or `deg`)
- `--output_root`: Output dataset root (optional)
- `--overwrite`: Overwrite existing EE-pose fields if present

**Output:**

- Single-arm: 8 dimensions (position xyz + quaternion xyzw + gripper)
- Dual-arm: 16 dimensions (left arm 8D + right arm 8D)

### 4.4 Add PointCloud

Add pointcloud to existing datasets offline (result in `XYZRGB`):

```bash
python scripts/utils/process_parquet_to_pc.py \
    --dataset_root Datasets/record/fold_garment/001 \
    --num_points 4096
```

**Parameters:**

- `--dataset_root`: Dataset root path
- `--num_points`: Number of sampled points in pointcloud

**Output:**

- Create `pointclouds` folder which contains all the episode folders, `episode_000` folder stores pointcloud for each frame, like `frame_000000.npz`

### 4.5 Replay Dataset

Replay recorded datasets for visualization, verification, or data augmentation. Supports joint angle actions or end-effector pose control (via IK).

**Example Command:**

```bash
python -m scripts.dataset_sim replay \
    --task LeHome-BiSO101-Direct-Garment-v0 \
    --dataset_root Datasets/example/fold_garment/001 \
    --num_replays 1 \
    --disable_depth \
    --device cpu \
    --enable_cameras
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--task` | str | `None` | Task environment name; required in practice because replay calls `parse_env_cfg(args.task, ...)` |
| `--dataset_root` | str | required | Input dataset directory |
| `--output_root` | str | `None` | Output directory for saved replay episodes |
| `--num_replays` | int | `1` | Number of replays per episode |
| `--save_successful_only` | flag | `False` | Only save episodes that achieve success |
| `--start_episode` | int | `0` | Starting episode index (inclusive) |
| `--end_episode` | int | `None` | Ending episode index |
| `--step_hz` | int | `60` | Replay stepping rate |
| `--use_random_seed` | flag | `False` | Use random seed path when supported by environment config |
| `--seed` | int | `42` | Fixed random seed (ignored when `--use_random_seed` is set) |
| `--task_description` | str | empty string | Task description string |
| `--disable_depth` | flag | `False` | Disable depth observation (faster replay) |
| `--use_ee_pose` | flag | `False` | Use `action.ee_pose` control (Cartesian space, converted via IK) |
| `--ee_urdf_path` | str | `Assets/robots/so101_new_calib.urdf` | URDF file path (required for `--use_ee_pose`) |
| `--ee_state_unit` | str | `rad` | Joint angle unit (`rad` or `deg`) |
| `--device` | str | `cuda:0` (AppLauncher default) | Simulator device (`cpu`, `cuda`, `cuda:N`) |

**Notes:**

- Pure replay mode: omit `--output_root` to replay without saving
- Data augmentation mode: use `--num_replays N` with `--save_successful_only` to keep only successful trajectories
- End-effector replay requires `action.ee_pose` in the dataset (recorded online or added via `augment`)

### 4.6 Remove Features

Remove specific features from datasets (e.g., depth maps to reduce storage):

**Remove a single feature:**

```bash
lerobot-edit-dataset \
    --repo_id fold_garment/001 \
    --root Datasets/record/fold_garment/001 \
    --new_repo_id fold_garment/001_no_depth \
    --operation.type remove_feature \
    --operation.feature_names "['observation.top_depth']"
```

**Batch remove features from multiple datasets:**

```bash
for i in {001..003}; do
    lerobot-edit-dataset \
        --repo_id fold_garment/$i \
        --root Datasets/record/fold_garment/$i \
        --new_repo_id fold_garment/${i}_no_depth \
        --operation.type remove_feature \
        --operation.feature_names "['observation.top_depth']"
done
```

**Parameters:**

- `--repo_id`: Dataset identifier. Used as a label in metadata and for downloading from HuggingFace Hub if local files are not found. When `--root` is specified, it doesn't affect the actual file path.
- `--root`: Dataset root directory path (relative or absolute). This determines where the dataset files are actually located.
- `--new_repo_id`: New dataset identifier (optional, if omitted, original dataset is renamed to `_old`)
- `--operation.type`: Operation type, use `remove_feature`
- `--operation.feature_names`: List of feature names to remove (Python list format)

**Notes:**

- Cannot remove required features: `timestamp`, `frame_index`, `episode_index`, `index`, `task_index`
- If `--new_repo_id` is specified, original dataset remains unchanged
- Multiple features can be removed: `"['feature1', 'feature2']"`

## 5. Dataset Merging

Merge multiple datasets collected from different sessions into a single unified dataset.

**Using LeHome merge command:**

```bash
python -m scripts.dataset merge \
    --source_roots "['Datasets/record/fold_garment/001', 'Datasets/record/fold_garment/002']" \
    --output_root Datasets/record/fold_garment_merged \
    --output_repo_id fold_garment_merged
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--source_roots` | required | Python-list string of dataset roots |
| `--output_root` | required | Output dataset root |
| `--output_repo_id` | `merged_dataset` | Output repo ID |
| `--merge_custom_meta` | `True` | Merge `episode_metadata.json` files |

**Notes:**

- All source datasets must have **identical feature structures** (except for removed features)
- Merged dataset contains all episodes from source datasets in sequential order
- Video files are automatically split into multiple files based on size limit (default: 200MB)
- Data files are automatically split into multiple files based on size limit (default: 100MB)
- The merge path also merges `meta/episode_metadata.json` and offsets episode indices automatically

## 6. Dataset Format

### 6.1 Data Features

The recorder builds the following LeRobot feature schema.

| Feature | Dual-Arm Shape | Single-Arm Shape | When Present |
|---------|----------------|------------------|--------------|
| `observation.state` | `(12,)` | `(6,)` | Always |
| `action` | `(12,)` | `(6,)` | Always |
| `observation.images.top_rgb` | `(480, 640, 3)` | `(480, 640, 3)` | Always |
| `observation.images.left_rgb` | `(480, 640, 3)` | - | Dual-arm tasks |
| `observation.images.right_rgb` | `(480, 640, 3)` | - | Dual-arm tasks |
| `observation.images.wrist_rgb` | - | `(480, 640, 3)` | Single-arm tasks when a wrist camera exists in the environment |
| `observation.top_depth` | `(480, 640)` | `(480, 640)` | Unless `--disable_depth` is set |
| `observation.ee_pose` | `(16,)` | `(8,)` | When `--record_ee_pose` is enabled or after offline augmentation |
| `action.ee_pose` | `(16,)` | `(8,)` | When `--record_ee_pose` is enabled or after offline augmentation |
| `task` | string | string | Written by the current recording loop from `--task_description` |

Depth storage details verified in the current code:

- stored dtype: `uint16`
- stored unit: `millimeters`
- storage conversion: `np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)`
- decode conversion: `depth_meters = uint16_value / 1000.0`
- recorder metadata stores the top-camera intrinsics and camera/base/world transforms under `meta/info.json -> features.observation.top_depth.info.camera`

### 6.2 Metadata

The `meta/` directory contains several metadata files:

- **`info.json`**: Dataset metadata including total episodes, total frames, feature definitions, FPS, and dataset configuration
- **`episode_metadata.json`**: Custom metadata file containing initial object pose information for each episode, grouped by `variant_name`:

  The following example uses a `fold_garment` dataset entry, where `variant_name` is `Tops` and the recorded object key is `Garment`.

  ```json
  {
    "Tops": {
      "0": {
        "object_initial_pose": {
          "Garment": {
            "trans": [...],
            "rot": [...],
            "scale": [...]
          }
        }
      },
      "1": {
        "object_initial_pose": {...}
      }
    }
  }
  ```

- **`stats.json`**: Statistical information about the dataset (for example mean/std or min/max for numeric features)
- **`tasks.parquet`**: Task descriptions for each episode in Parquet format
- **`episodes/`**: Episode-level metadata stored in Parquet format, organized by chunks

The current recorder appends the initial object pose under `episode_metadata.json`, grouped by `variant_name` when the environment provides it. The LeHome merge command also merges this file and offsets episode indices automatically.

## 7. Best Practices

- Prepare `Assets/` before running recording, replay, or evaluation commands.
- Use `--enable_cameras` for the workflows documented here.
- Prefer SO101 leader teleoperation for data collection.
- Consistent Task Description: Use the same `--task_description` for similar tasks to enable better filtering during training.
- Exit mid-recording with `ESC` if you need to stop data collection before finishing.
- Episode Quality: Use the `D` key to discard low-quality episodes during recording.
- Backup: Always backup your datasets before merging or processing.
