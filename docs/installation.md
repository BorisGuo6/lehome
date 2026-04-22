# UV installation guide

This guide provides step-by-step instructions for installing the LeHome environment using `uv`.

## Prerequisites

- Python 3.11
- [uv](https://github.com/astral-sh/uv) package manager
- GPU driver and CUDA supporting IsaacSim5.1.0.

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/lehome-official/lehome.git
cd lehome
```

### 2. Install Dependencies with uv

```bash
uv sync
```

This will create a virtual environment and install all required dependencies.

### 3. Clone and Configure IsaacLab

```bash
cd third_party
git clone https://github.com/lehome-official/IsaacLab.git
cd ..
```

### 4. Install IsaacLab

Activate the virtual environment and install IsaacLab:

```bash
source .venv/bin/activate
./third_party/IsaacLab/isaaclab.sh -i none
```

### 5. Install LeHome Package

Finally, install the LeHome package in development mode:

```bash
uv pip install -e ./source/lehome
```

This exposes the local `lehome` package and task registrations to the CLI entry points used throughout the docs.

## Optional Server Dependencies

If you are using a server, please download the system dependencies.

```bash
    #step 1
    apt update
    apt install -y \
    libglu1-mesa \
    libgl1 \
    libegl1 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libxext6 \
    libx11-6
    #step 2
    export __GLX_VENDOR_LIBRARY_NAME=nvidia
```

## Next Steps

Now that you have installed the environment, you can:

- [Dataset Collection and Processing Guide](datasets.md)
- [Training Guide](training.md)
- [Policy Evaluation Guide](policy_eval.md)
- [Scene Deactivation Guide](scene_deactivation.md)
- [Back to README](../README.md)
