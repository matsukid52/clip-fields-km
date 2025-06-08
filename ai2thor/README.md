# AI2-THOR Interactive Recorder for clip-fields

This directory contains a pipeline to interactively record AI2-THOR scenes and generate `.r3d` files compatible with the `clip-fields` training model.

## Overview

The `interactive_recorder.py` script launches an AI2-THOR environment in an interactive window. You can navigate the scene using your keyboard. When you start recording, the script captures RGB-D data, camera poses, and intrinsics at 30fps. Once you stop recording, this data is packaged into a single `.r3d` file.

This `.r3d` file is a renamed `.zip` archive that follows the exact structure required by the `R3DSemanticDataset` in the `clip-fields` project, allowing you to seamlessly use it for training.

## Files

- `interactive_recorder.py`: The main script to run the interactive session.
- `requirements.txt`: A list of Python dependencies required for the script.
- `README.md`: This documentation.

## Setup

1.  **Install Dependencies**: First, install the necessary Python packages. Navigate to this directory in your terminal and run:
    ```bash
    pip install -r requirements.txt
    ```

2.  **X Server (Linux-Only)**: AI2-THOR requires a running X display server on Linux. If you are running on a headless server, you might need to use `Xvfb`.

## How to Use

1.  **Run the Recorder**: Start the interactive session by running the script from this directory:
    ```bash
    python interactive_recorder.py
    ```
    You can also specify a different scene and resolution:
    ```bash
    python interactive_recorder.py --scene FloorPlan_Train1_1 --width 800 --height 600
    ```
    A Pygame window will open, displaying the AI2-THOR scene.

2.  **Controls**:
    - **W, A, S, D**: Move forward, left, backward, and right.
    - **Arrow Keys**: Turn left/right and look up/down.
    - **R**: Press once to **start** recording. A "RECORDING" indicator will appear.
    - **R**: Press again to **stop** recording. The script will then process the captured frames and save the `.r3d` file.
    - **Q**: Quit the application.

3.  **Output**:
    - After you stop recording, a new file named `{scene_name}_{timestamp}.r3d` will be created in the current directory (e.g., `FloorPlan1_physics_20231027_103000.r3d`).
    - This file is now ready to be used with the `clip-fields` training script.

## Training with Your Custom `.r3d` File

Once you have your recorded `my_scene.r3d` file, you can train `clip-fields` on it just like you would with the `nyu.r3d` example:

```bash
python train.py dataset_path=path/to/my_scene.r3d
```

This pipeline provides a complete workflow for generating custom training data for `clip-fields` from any of the hundreds of scenes available in AI2-THOR. 