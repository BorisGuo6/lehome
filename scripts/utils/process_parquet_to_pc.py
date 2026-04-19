import sys
import os
import argparse
import glob
import json
import numpy as np
import pyarrow.parquet as pq
import imageio.v3 as iio
from pathlib import Path
from tqdm import tqdm
from lehome.utils.depth_to_pointcloud import generate_pointcloud_from_data


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))


def load_camera_kwargs_from_dataset(dataset_path: Path) -> dict:
    """Load task-specific camera reprojection parameters from dataset metadata."""
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        print(
            f"Warning: {info_path} not found. "
            "Using built-in fallback pointcloud defaults, which may not match this task."
        )
        return {}

    try:
        info = json.loads(info_path.read_text())
    except Exception as e:
        print(
            f"Warning: Failed to read {info_path}: {e}. "
            "Using built-in fallback pointcloud defaults, which may not match this task."
        )
        return {}

    camera_info = (
        info.get("features", {})
        .get("observation.top_depth", {})
        .get("info", {})
        .get("camera")
    )
    if not camera_info:
        print(
            "Warning: No camera metadata found under "
            "'features.observation.top_depth.info.camera'. "
            "Using built-in fallback pointcloud defaults, which may not match this task."
        )
        return {}

    intrinsics = camera_info.get("intrinsics", {})
    camera_to_base = camera_info.get("camera_to_base", {})
    base_to_world = camera_info.get("base_to_world", {})

    required = {
        "intrinsics.fx": intrinsics.get("fx"),
        "intrinsics.fy": intrinsics.get("fy"),
        "intrinsics.cx": intrinsics.get("cx"),
        "intrinsics.cy": intrinsics.get("cy"),
        "camera_to_base.translation": camera_to_base.get("translation"),
        "camera_to_base.rotation_wxyz": camera_to_base.get("rotation_wxyz"),
        "base_to_world.translation": base_to_world.get("translation"),
        "base_to_world.rotation_wxyz": base_to_world.get("rotation_wxyz"),
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        print(
            f"Warning: Camera metadata in {info_path} is incomplete (missing: {missing}). "
            "Using built-in fallback pointcloud defaults, which may not match this task."
        )
        return {}

    print(f"Loaded camera metadata from {info_path}")
    return {
        "camera_fx": float(intrinsics["fx"]),
        "camera_fy": float(intrinsics["fy"]),
        "camera_cx": float(intrinsics["cx"]),
        "camera_cy": float(intrinsics["cy"]),
        "cam_to_base_quat_wxyz": list(camera_to_base["rotation_wxyz"]),
        "cam_to_base_translation": np.asarray(
            camera_to_base["translation"], dtype=np.float32
        ),
        "world_base_pos": np.asarray(base_to_world["translation"], dtype=np.float32),
        "world_base_rot_wxyz": list(base_to_world["rotation_wxyz"]),
    }


def get_args():
    parser = argparse.ArgumentParser(description="Process Parquet depth and MP4 RGB into Pointclouds.")
    parser.add_argument(
        "--dataset_root", 
        type=str, 
        required=True, 
        help="Path to the dataset root, e.g., /path/to/lehome/Datasets/record/001"
    )
    parser.add_argument(
        "--num_points", 
        type=int, 
        default=4096, 
        help="Number of points per pointcloud"
    )
    return parser.parse_args()

def main():
    args = get_args()
    dataset_path = Path(args.dataset_root)
    camera_kwargs = load_camera_kwargs_from_dataset(dataset_path)

    # ==========================================
    # 2. Locate RGB Video File
    # ==========================================
    # Assumes this video contains frames for ALL episodes sequentially
    video_path = dataset_path / "videos" / "observation.images.top_rgb" / "chunk-000" / "file-000.mp4"
    
    if not video_path.exists():
        print(f"Error: Video file not found at {video_path}")
        return

    print(f"Found Master Video: {video_path}")

    # ==========================================
    # 3. Locate Parquet Files
    # ==========================================
    parquet_dir = dataset_path / "data" / "chunk-000"
    if not parquet_dir.exists():
        print(f"Error: Parquet directory not found at {parquet_dir}")
        return

    parquet_files = sorted(glob.glob(str(parquet_dir / "file-*.parquet")))
    
    if not parquet_files:
        print("No parquet files found.")
        return

    print(f"Found {len(parquet_files)} episode parquet files.")

    # ==========================================
    # 4. Initialize Video Stream Iterator (using imiter)
    # ==========================================
    video_reader = iio.imiter(str(video_path), plugin="pyav")
    video_iterator = iter(video_reader)

    # ==========================================
    # 5. Processing Loop
    # ==========================================
    total_frames_processed = 0

    for ep_idx, pq_file in enumerate(tqdm(parquet_files, desc="Processing Episodes")):
        try:
            parquet_file = pq.ParquetFile(pq_file)
            num_frames_in_episode = parquet_file.metadata.num_rows
        except Exception as e:
            print(f"Error reading parquet {pq_file}: {e}")
            continue

        # --- Prepare Output Path ---
        output_dir = dataset_path / "pointclouds" / f"episode_{ep_idx:03d}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Process by Frame ---
        frame_pbar = tqdm(
            total=num_frames_in_episode,
            desc=f"Episode {ep_idx:03d} Frames",
            leave=False,
        )
        frame_idx = 0

        try:
            for batch in parquet_file.iter_batches(
                columns=["observation.top_depth"], batch_size=32
            ):
                depth_column = batch.column(0)
                for row_idx in range(len(depth_column)):
                    try:
                        rgb_frame = next(video_iterator)

                        depth_frame = np.array(
                            depth_column[row_idx].as_py(), dtype=np.float32
                        )

                        # Make Sure Shape is (480, 640)
                        if depth_frame.ndim == 1:
                            if depth_frame.size == 480 * 640:
                                depth_frame = depth_frame.reshape((480, 640))
                            else:
                                print(
                                    f"Error: Depth frame size {depth_frame.size} does not match 480x640"
                                )
                                continue

                        # Since the depth information was stored in uint16 and in mm
                        depth_frame = depth_frame.astype(np.float32) / 1000.0

                        pointclouds_with_color = generate_pointcloud_from_data(
                            rgb_image=rgb_frame,
                            depth_image=depth_frame,
                            num_points=args.num_points,
                            use_fps=True,
                            **camera_kwargs,
                        )

                        save_path = output_dir / f"frame_{frame_idx:06d}.npz"
                        np.savez_compressed(str(save_path), pointcloud=pointclouds_with_color)

                        total_frames_processed += 1
                    except StopIteration:
                        print(
                            f"Error: Video ended unexpectedly at Episode {ep_idx}, Frame {frame_idx}!"
                        )
                        raise
                    except Exception as e:
                        print(f"Error processing Ep {ep_idx} Frame {frame_idx}: {e}")
                    finally:
                        frame_idx += 1
                        frame_pbar.update(1)
        except StopIteration:
            pass
        finally:
            frame_pbar.close()

    print(f"\nAll done! Processed {total_frames_processed} frames across {len(parquet_files)} episodes.")

if __name__ == "__main__":
    main()
    
