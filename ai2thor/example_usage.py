#!/usr/bin/env python3
"""
Example usage of the AI2THOR R3D Recorder

This script demonstrates different ways to use the AI2THOR R3D recorder
to create .r3d files for training clip-fields models.
"""

from record_r3d import AI2THORR3DRecorder
import sys
import os

def example_basic_recording():
    """Basic example of recording a scene."""
    print("=== Basic Recording Example ===")
    
    # Create recorder for a kitchen scene
    recorder = AI2THORR3DRecorder(
        scene_name="FloorPlan1_1",  # Kitchen scene
        width=640,
        height=480,
        fov=60.0
    )
    
    try:
        # Record 50 frames
        num_frames = recorder.navigate_and_record(num_frames=50)
        
        if num_frames > 0:
            # Save the recording
            recorder.save_r3d_format("kitchen_recording.r3d")
            print(f"✓ Successfully recorded {num_frames} frames from kitchen scene")
        else:
            print("✗ Failed to record any frames")
            
    finally:
        recorder.cleanup()

def example_high_res_recording():
    """Example of high-resolution recording."""
    print("\n=== High Resolution Recording Example ===")
    
    # Create recorder with higher resolution
    recorder = AI2THORR3DRecorder(
        scene_name="FloorPlan2_2",  # Living room scene
        width=1280,
        height=720,
        fov=70.0
    )
    
    try:
        # Record fewer frames due to higher resolution
        num_frames = recorder.navigate_and_record(num_frames=30)
        
        if num_frames > 0:
            recorder.save_r3d_format("living_room_hd.r3d")
            print(f"✓ Successfully recorded {num_frames} HD frames from living room")
        else:
            print("✗ Failed to record any frames")
            
    finally:
        recorder.cleanup()

def example_controlled_movement():
    """Example with controlled movement patterns."""
    print("\n=== Controlled Movement Example ===")
    
    recorder = AI2THORR3DRecorder(
        scene_name="FloorPlan3_3",  # Bedroom scene
        width=800,
        height=600,
        fov=65.0
    )
    
    try:
        # Use only rotation and looking actions for smoother recording
        movement_actions = ["RotateLeft", "RotateRight", "LookUp", "LookDown"]
        
        num_frames = recorder.navigate_and_record(
            num_frames=40,
            movement_actions=movement_actions
        )
        
        if num_frames > 0:
            recorder.save_r3d_format("bedroom_smooth.r3d")
            print(f"✓ Successfully recorded {num_frames} frames with smooth movements")
        else:
            print("✗ Failed to record any frames")
            
    finally:
        recorder.cleanup()

def demonstrate_clip_fields_integration():
    """Show how to integrate with clip-fields training."""
    print("\n=== Clip-Fields Integration Example ===")
    
    # This would be how you'd use the recorded data with clip-fields
    print("""
After recording .r3d files, you can use them with clip-fields like this:

1. In your training script:
   ```python
   from dataloaders.record3d import R3DSemanticDataset
   
   # Load the recorded scene
   dataset = R3DSemanticDataset('kitchen_recording.r3d')
   
   # Use with your training pipeline
   for i in range(len(dataset)):
       sample = dataset[i]
       rgb = sample['rgb']           # RGB image
       depth = sample['depth']       # Depth map  
       xyz = sample['xyz_position']  # 3D coordinates
       conf = sample['conf']         # Confidence map
   ```

2. The dataset integrates seamlessly with the existing clip-fields pipeline:
   - RGB images for visual features
   - Depth maps for geometric understanding
   - XYZ coordinates for 3D spatial reasoning
   - Camera poses for view synthesis
   
3. You can record multiple scenes and combine them:
   ```python
   scenes = ['kitchen_recording.r3d', 'living_room_hd.r3d', 'bedroom_smooth.r3d']
   datasets = [R3DSemanticDataset(scene) for scene in scenes]
   ```
""")

def main():
    """Run all examples."""
    print("AI2THOR R3D Recording Examples")
    print("="*50)
    
    try:
        # Run examples
        example_basic_recording()
        example_high_res_recording() 
        example_controlled_movement()
        demonstrate_clip_fields_integration()
        
        print("\n" + "="*50)
        print("All examples completed successfully!")
        print("\nGenerated files:")
        for filename in ["kitchen_recording.r3d", "living_room_hd.r3d", "bedroom_smooth.r3d"]:
            if os.path.exists(filename):
                size_mb = os.path.getsize(filename) / (1024 * 1024)
                print(f"  {filename} ({size_mb:.1f} MB)")
        
        print(f"\nNext steps:")
        print(f"1. Install clip-fields dependencies if not already done")
        print(f"2. Use the generated .r3d files with R3DSemanticDataset")
        print(f"3. Train your clip-fields model with the recorded scenes")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 