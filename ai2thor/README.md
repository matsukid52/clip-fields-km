# AI2THOR R3D Recording Scripts

This directory contains scripts to record .r3d files using AI2THOR (specifically RoboTHOR) scenes for training clip-fields models.

## Overview

The `.r3d` format is typically created by the Record3D iOS app, containing RGB-D sequences with camera poses. This script simulates that format using AI2THOR's rich 3D environments, allowing you to generate training data for clip-fields without needing physical recording devices.

## Files

- `record_r3d.py` - Main recording script with the `AI2THORR3DRecorder` class
- `example_usage.py` - Examples showing different recording configurations
- `README.md` - This documentation file

## Requirements

Make sure you have the following dependencies installed:

```bash
pip install ai2thor numpy pillow tqdm
```

Optional (for real compression compatibility):
```bash
pip install lzfse  # May require additional system dependencies
```

## Basic Usage

### Command Line Interface

Record a simple scene with default settings:
```bash
cd ai2thor
python record_r3d.py
```

Record a specific scene with custom parameters:
```bash
python record_r3d.py --scene FloorPlan2_1 --frames 200 --output my_recording.r3d
```

Full command line options:
```bash
python record_r3d.py --help
```

### Programmatic Usage

```python
from record_r3d import AI2THORR3DRecorder

# Create recorder
recorder = AI2THORR3DRecorder(
    scene_name="FloorPlan1_1",
    width=640,
    height=480,
    fov=60.0
)

# Record data
num_frames = recorder.navigate_and_record(num_frames=100)

# Save in .r3d format
if num_frames > 0:
    recorder.save_r3d_format("my_scene.r3d")

# Clean up
recorder.cleanup()
```

## How It Works

### 1. Scene Setup
- Initializes an AI2THOR controller with the specified scene
- Configures camera parameters (resolution, field of view)
- Enables depth rendering and other required features

### 2. Data Capture
- Navigates through the scene using random movements and rotations
- Captures RGB images, depth maps, and camera poses at each step
- Simulates confidence maps (since AI2THOR doesn't provide them)
- Periodically teleports to new locations to explore more of the scene

### 3. .r3d Format Generation
- Packages all captured data into a zip file with the .r3d extension
- Creates metadata compatible with the `R3DSemanticDataset` loader
- Compresses depth and confidence data (using LZFSE if available)
- Stores RGB images as JPEG files

### 4. Data Structure
The generated .r3d file contains:
```
metadata          # JSON with camera parameters, poses, scene info
rgbd/0.jpg        # RGB image for frame 0
rgbd/0.depth      # Compressed depth data for frame 0  
rgbd/0.conf       # Compressed confidence data for frame 0
rgbd/1.jpg        # RGB image for frame 1
...
```

## Integration with Clip-Fields

Once you have recorded .r3d files, you can use them directly with the existing clip-fields pipeline:

```python
from dataloaders.record3d import R3DSemanticDataset

# Load your recorded scene
dataset = R3DSemanticDataset('path/to/your/recording.r3d')

# Use in training
for i in range(len(dataset)):
    sample = dataset[i]
    rgb = sample['rgb']           # RGB image (H, W, 3)
    depth = sample['depth']       # Depth map (H, W)
    xyz = sample['xyz_position']  # 3D coordinates (N, 3) 
    conf = sample['conf']         # Confidence map (H, W)
```

## Available Scenes

AI2THOR provides many scene types:

### Kitchen Scenes
- `FloorPlan1_1` through `FloorPlan30_5` - Various kitchen layouts

### Living Room Scenes  
- `FloorPlan201_1` through `FloorPlan230_4` - Living room environments

### Bedroom Scenes
- `FloorPlan301_1` through `FloorPlan330_5` - Bedroom layouts

### Bathroom Scenes
- `FloorPlan401_1` through `FloorPlan430_4` - Bathroom environments

### RoboTHOR Scenes
- `FloorPlan_Train1_1` through `FloorPlan_Train12_5` - Training scenes
- `FloorPlan_Val1_1` through `FloorPlan_Val3_5` - Validation scenes

## Tips for Better Recordings

1. **Scene Selection**: Different scenes provide different object types and layouts
2. **Resolution**: Higher resolution provides more detail but takes longer to process
3. **Frame Count**: More frames give better coverage but larger file sizes
4. **Movement Patterns**: Controlled movements (rotation only) can be smoother
5. **Multiple Recordings**: Record several scenes to increase data diversity

## Troubleshooting

### Common Issues

**"No depth image available"**
- Make sure `renderDepthImage=True` in the controller initialization
- Some actions might fail - the script will retry automatically

**Slow recording**
- Lower the resolution or reduce the number of frames
- AI2THOR can be slower than real-time, especially with high quality settings

**Import errors**
- Make sure ai2thor is properly installed: `pip install ai2thor`
- The script will work without lzfse, just with simulated compression

**Empty recordings**
- Some scenes might have movement restrictions
- Try different scenes or movement patterns
- The script automatically teleports to explore more areas

### Performance Tips

- Use lower resolution for faster recording (640x480 vs 1280x720)
- Reduce visual quality if speed is more important than fidelity
- Record multiple smaller sequences rather than one large one

## Differences from Real Record3D

This simulation aims to be compatible with clip-fields' existing .r3d loader, but there are some differences from real Record3D files:

1. **Confidence Maps**: Simulated based on depth values rather than actual sensor confidence
2. **Compression**: May use different compression if lzfse is not available
3. **Camera Movement**: Discrete movements rather than smooth camera motion
4. **Scene Content**: Synthetic 3D environments rather than real-world captures

These differences should not affect the clip-fields training pipeline, as the data format and structure remain compatible. 