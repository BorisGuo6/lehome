# Training Guide

This guide covers how to train policies for LeHome, including the provided LeRobot policy configs, feature selection, and training parameters.

## Table of Contents

- [Training Guide](#training-guide)
  - [Table of Contents](#table-of-contents)
  - [1. Quick Start](#1-quick-start)
  - [2. Training with Official Policies](#2-training-with-official-policies)
    - [2.1 Available Policies](#21-available-policies)
    - [2.2 Basic Training Command](#22-basic-training-command)
    - [2.3 Configuration File Structure](#23-configuration-file-structure)
    - [2.4 Dataset Features](#24-dataset-features)
    - [2.5 Feature Selection](#25-feature-selection)
      - [Feature Types](#feature-types)
      - [Example Feature Combinations](#example-feature-combinations)
      - [Using Partial Cameras](#using-partial-cameras)
    - [2.6 Training Parameters](#26-training-parameters)
      - [Dataset Configuration](#dataset-configuration)
      - [Policy Configuration](#policy-configuration)
      - [Training Hyperparameters](#training-hyperparameters)
      - [Output Configuration](#output-configuration)
      - [WandB Configuration](#wandb-configuration)
  - [3. Additional Resources](#3-additional-resources)

---

## 1. Quick Start

Train a policy using one of the pre-configured training files:

```bash
lerobot-train --config_path=configs/train_act.yaml
```

**Available config files:**

- `configs/train_act.yaml` - ACT policy
- `configs/train_dp.yaml` - Diffusion Policy
- `configs/train_smolvla.yaml` - SmolVLA policy

---

## 2. Training with Official Policies

### 2.1 Available Policies

LeHome currently provides configuration files for the following policies:

| Policy | Type | Description | Config File |
|--------|------|-------------|-------------|
| `act` | Imitation Learning | Action Chunking Transformer | `configs/train_act.yaml` |
| `diffusion` | Imitation Learning | Diffusion Policy | `configs/train_dp.yaml` |
| `smolvla` | Vision-Language-Action | Small Vision-Language-Action model | `configs/train_smolvla.yaml` |

> **Note**: LeRobot supports additional policies (π0, π0.5, GR00T, X-VLA), but configuration files for these are not provided in this repository. You can create custom configuration files following the [LeRobot documentation](https://huggingface.co/docs/lerobot) or use the above three baseline policies.

### 2.2 Basic Training Command

The recommended way to train is using a configuration file:

```bash
lerobot-train --config_path=path/to/your/config.yaml
```

> **Note:** Using configuration files allows you to explicitly specify which features to use for training.

### 2.3 Configuration File Structure

A typical training configuration file looks like this:

```yaml
dataset:
  repo_id: <repo_name>
  root: Datasets/<dataset_name>

policy:
  type: <policy_type>
  device: cuda
  push_to_hub: false

  input_features:
    observation.state:
      type: STATE
      shape: [12]
    observation.images.top_rgb:
      type: VISUAL
      shape: [3, 480, 640]
    observation.images.left_rgb:
      type: VISUAL
      shape: [3, 480, 640]
    observation.images.right_rgb:
      type: VISUAL
      shape: [3, 480, 640]

output_features:
    action:
      type: ACTION
      shape: [12]

output_dir: outputs/train/<output_name>
batch_size: 16
steps: 30000
save_freq: 5000
log_freq: 1000

wandb:
  enable: false
```

**Key sections:**

- `dataset`: Specifies the dataset location
- `policy`: Defines policy type, device, and input/output features
- `output_dir`: Where to save checkpoints and logs
- `training hyperparameters`: `batch_size`, `steps`, `save_freq`, `log_freq`
- `wandb`: Weights & Biases logging configuration, enable if needed

### 2.4 Dataset Features

LeHome datasets can include the following features, depending on whether the task is single-arm or dual-arm and whether optional recording flags were enabled:

| Feature | Dual-Arm Shape | Single-Arm Shape | Description |
|---------|----------------|------------------|-------------|
| `observation.state` | `(12,)` | `(6,)` | Joint positions |
| `action` | `(12,)` | `(6,)` | Joint actions |
| `observation.images.top_rgb` | `(480, 640, 3)` | `(480, 640, 3)` | Top camera RGB image |
| `observation.images.left_rgb` | `(480, 640, 3)` | - | Left RGB image for dual-arm tasks |
| `observation.images.right_rgb` | `(480, 640, 3)` | - | Right RGB image for dual-arm tasks |
| `observation.images.wrist_rgb` | - | `(480, 640, 3)` | Wrist RGB image for single-arm tasks when the environment provides it |
| `observation.top_depth` | `(480, 640)` | `(480, 640)` | Top camera depth map stored as `uint16` millimeters |
| `observation.ee_pose` | `(16,)` | `(8,)` | End-effector observation pose, available when recorded or added offline |
| `action.ee_pose` | `(16,)` | `(8,)` | End-effector action pose, available when recorded or added offline |
| `task` | `string` | `string` | Task description |

**Notes:**

- The shipped training configs currently use joint-space state/action plus RGB cameras.
- For single-arm tasks, replace dual-arm visual inputs (`left_rgb`, `right_rgb`) with the actual single-arm camera keys present in the dataset, typically `wrist_rgb`.
- `observation.top_depth` is stored in the dataset as a 2D `uint16` depth image, but when used as a training feature it is typically configured as `shape: [1, 480, 640]`.
- Using `observation.ee_pose` and `action.ee_pose` is possible, but joint-space control (`observation.state` and `action`) is usually the safer default on SO101 hardware because IK-based pipelines are more sensitive.

### 2.5 Feature Selection

You can flexibly select which features to use for training by specifying them in the `input_features` and `output_features` sections.

#### Feature Types

When configuring features, note that:

- **RGB images** (`observation.images.*_rgb`) use `type: VISUAL`
- **Depth maps** (`observation.top_depth`) use `type: STATE`
- **Joint states and end-effector poses** (`observation.state`, `observation.ee_pose`) use `type: STATE`
- **Actions** (`action`, `action.ee_pose`) use `type: ACTION`

> **Note:** `observation.top_depth` is configured as `STATE` because LeRobot's visual feature consistency validation only checks features explicitly marked as `VISUAL` (RGB images). Using `STATE` for depth maps allows more flexible configuration.

#### Example Feature Combinations

The following feature combinations align with the dataset schema used in this repository.

**Combination 1: State + RGB Cameras**

This is the layout used by the shipped training YAML files.

```yaml
input_features:
  observation.state:
    type: STATE
    shape: [12]
  observation.images.top_rgb:
    type: VISUAL
    shape: [3, 480, 640]
  observation.images.left_rgb:
    type: VISUAL
    shape: [3, 480, 640]
  observation.images.right_rgb:
    type: VISUAL
    shape: [3, 480, 640]
```

**Combination 2: State + RGB Cameras + Depth**

Use this when your dataset contains `observation.top_depth` and your policy should consume depth explicitly.

```yaml
input_features:
  observation.state:
    type: STATE
    shape: [12]
  observation.images.top_rgb:
    type: VISUAL
    shape: [3, 480, 640]
  observation.images.left_rgb:
    type: VISUAL
    shape: [3, 480, 640]
  observation.images.right_rgb:
    type: VISUAL
    shape: [3, 480, 640]
  observation.top_depth:
    type: STATE
    shape: [1, 480, 640]
```

**Combination 3: End-Effector Pose + RGB Cameras + Depth**

This layout is possible if your dataset contains `observation.ee_pose`, `action.ee_pose`, and `observation.top_depth`.

> **Not Recommended:** This combination relies on end-effector pose features, which are more sensitive to IK quality and calibration. Prefer joint-space control unless you specifically need Cartesian-space supervision.

```yaml
input_features:
  observation.ee_pose:
    type: STATE
    shape: [16]
  observation.images.top_rgb:
    type: VISUAL
    shape: [3, 480, 640]
  observation.images.left_rgb:
    type: VISUAL
    shape: [3, 480, 640]
  observation.images.right_rgb:
    type: VISUAL
    shape: [3, 480, 640]
  observation.top_depth:
    type: STATE
    shape: [1, 480, 640]
```

For single-arm tasks, adapt the dimensions accordingly:

- `observation.state`: `[6]`
- `action`: `[6]`
- `observation.ee_pose`: `[8]`
- `action.ee_pose`: `[8]`
- replace `left_rgb` and `right_rgb` with `wrist_rgb` when applicable

#### Using Partial Cameras

If you want to train with only a subset of cameras, keep the following in mind:

- define only the camera features you actually want to use under `policy.input_features`
- make sure the feature names exactly match those recorded in the dataset
- for dual-arm datasets, it is valid to use only `top_rgb`, or `top_rgb` plus one side camera, as long as the config and the policy processor agree on the feature set
- the current repository does not ship a dedicated partial-camera training config, so the safest approach is to start from one of the provided YAML files and remove unused visual inputs carefully

### 2.6 Training Parameters

#### Dataset Configuration

```yaml
dataset:
  repo_id: <repo_name>
  root: Datasets/<dataset_name>
```

Notes:

- `dataset.root` points to the actual local dataset directory
- `dataset.repo_id` is a dataset identifier used by LeRobot; when `root` is set for local training, `root` is what determines the real file location
- before training on your own data, update both fields to match your dataset naming

#### Policy Configuration

```yaml
policy:
  type: <policy_type>
  device: cuda
  push_to_hub: false
```

Optional policy-specific fields used in the shipped configs:

```yaml
policy:
  type: diffusion
  device: cuda
  push_to_hub: false
  crop_shape: null
  crop_is_random: false
```

Notes:

- `policy.type` is `act`, `diffusion`, or `smolvla` in the shipped configs
- `policy.device` is set to `cuda` in all provided training YAML files
- `policy.push_to_hub` is disabled by default
- `crop_shape` and `crop_is_random` are only present in `configs/train_dp.yaml`

#### Training Hyperparameters

| Parameter | Description | Values used in this repository |
|-----------|-------------|--------------------------------|
| `batch_size` | Batch size for training | `16` for ACT and Diffusion, `32` for SmolVLA |
| `steps` | Total training steps | `30000` |
| `save_freq` | Checkpoint save frequency | `5000` |
| `log_freq` | Logging frequency | `1000` |
| `learning_rate` | Learning rate | not explicitly set in the shipped YAML files; inherited from policy defaults unless you add it |

#### Output Configuration

```yaml
output_dir: outputs/train/<output_name>
```

The shipped configs currently use:

- `outputs/train/act_so101_test`
- `outputs/train/dp_so101_test`
- `outputs/train/smolvla_so101_test`

Checkpoints are typically saved to:

- `{output_dir}/checkpoints/last/pretrained_model` - latest checkpoint
- `{output_dir}/checkpoints/step_{N}/pretrained_model` - periodic checkpoints

#### WandB Configuration

The shipped YAML files currently use:

```yaml
wandb:
  enable: false
```

If you want to enable Weights & Biases logging, you can extend it like this:

```yaml
wandb:
  enable: true
  project: my_project_name
  entity: my_username
```

---

## 3. Additional Resources

- [Dataset Collection and Processing Guide](datasets.md)
- [Policy Evaluation Guide](policy_eval.md)
- [Installation Guide](installation.md)
- [LeRobot Official Documentation](https://huggingface.co/docs/lerobot)
