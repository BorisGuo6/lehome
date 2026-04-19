# Policy Evaluation Guide

This guide covers evaluation of LeRobot policies in LeHome.

---

## Using LeRobot Policies

Evaluate trained LeRobot models (ACT, Diffusion Policy, SmolVLA):

```bash
python -m scripts.eval \
    --task <task_name> \
    --policy_type lerobot \
    --policy_path outputs/train/<output_name>/checkpoints/last/pretrained_model \
    --dataset_root Datasets/<dataset_name> \
    --task_description "<task_description>" \
    --num_episodes 5 \
    --enable_cameras \
    --device <sim_device>
```

> **Tip:** Choose `cpu`, `cuda`, or `cuda:N` for `<sim_device>` according to the task and runtime environment. `--enable_cameras` is required to see camera views in the GUI or record videos.

**Requirements:**

- `pretrained_model` directory or equivalent LeRobot checkpoint path
- training dataset metadata available under `--dataset_root`
- an explicit `--task` value registered in this repository

#### Common Options

| Parameter | Description | Default | Required For |
|-----------|-------------|---------|--------------|
| `--task` | Task ID registered in this repository | `None` | All |
| `--policy_type` | Policy type, use `lerobot` | `lerobot` | All |
| `--policy_path` | Path to the LeRobot model checkpoint | `None` | All |
| `--dataset_root` | Dataset root used for metadata loading | `None` | LeRobot |
| `--num_episodes` | Number of evaluation episodes | `5` | All |
| `--max_steps` | Maximum steps per episode | `600` | All |
| `--step_hz` | Evaluation stepping rate | `120` | All |
| `--task_description` | Task goal string | empty string | Recommended for language-conditioned policies |
| `--save_video` | Save evaluation videos | disabled | Optional |
| `--video_dir` | Video output directory | `outputs/eval_videos` | `--save_video` |
| `--save_datasets` | Save successful trajectories as a LeRobot dataset | disabled | Optional |
| `--eval_dataset_path` | Parent directory for saved evaluation datasets | `Datasets/eval` | `--save_datasets` |
| `--use_ee_pose` | Interpret policy outputs as EE pose and solve IK | disabled | Optional |
| `--ee_urdf_path` | URDF path for IK evaluation | `Assets/robots/so101_new_calib.urdf` | `--use_ee_pose` |
| `--use_random_seed` | Use env randomization seed path when supported | disabled | Optional |
| `--seed` | Fixed seed | `42` | Optional |
| `--enable_cameras` | Enable camera rendering | disabled | Recommended |
| `--device` | Simulator device: `cpu`, `cuda`, or `cuda:N` | `cuda:0` | All |
| `--headless` | Run without GUI | disabled | Optional |

**Parameter Descriptions:**

- `--policy_path` and `--dataset_root` are required for LeRobot evaluation.
- `--policy_type` should remain `lerobot` for the documented evaluation flow in this repository.

#### Current Behavior Notes

- the documented evaluation flow in this repository targets `--policy_type lerobot`
- `--device` controls the simulator device, not the policy inference device
- the current evaluation code loads the policy on `cuda` when `torch.cuda.is_available()` is true, otherwise `cpu`
- `--policy_path` and `--dataset_root` are mandatory for `--policy_type lerobot`
- `--num_envs` is parsed with default `1`, but is not currently applied to `env_cfg` in the evaluation path
- `--use_ee_pose` enables IK conversion before stepping the environment
- if `--save_datasets` is enabled and `--dataset_root` exists, the saved evaluation dataset inherits the source dataset schema and FPS
- success checking is task-driven and only works when the environment implements `_get_success()`

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| LeRobot fails to load | Ensure `--policy_path` points to `pretrained_model` dir or another valid LeRobot checkpoint and `--dataset_root` is correct |
| Action dimension error | Verify action shape: `(6)` for single-arm and `(12)` for dual-arm |
| No camera observations | Add `--enable_cameras` |
| No evaluation dataset saved | Enable `--save_datasets` and check `--eval_dataset_path` |

---

## Reference Files

- `scripts/eval.py` - evaluation entry point
- `scripts/utils/evaluation.py` - evaluation loop implementation
- `scripts/eval_policy/lerobot_policy.py` - built-in LeRobot policy wrapper

---

For detailed command-line arguments and repository-specific notes, see `scripts/utils/parser.py` and `scripts/utils/evaluation.py`.
