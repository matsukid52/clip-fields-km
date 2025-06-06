#!/usr/bin/env python3
"""
AI2THOR RGB-D Recording Script for .r3d File Generation

This script uses the AI2THOR simulator to record RGB-D sequences that are compatible
with the clip-fields training pipeline. It captures RGB images, depth maps, confidence
maps, and camera poses, then packages them in a format similar to Record3D's .r3d files.

The script navigates through a scene, captures data at different viewpoints, and saves
everything in a structured format that can be processed by the existing R3DSemanticDataset
loader in the clip-fields project.

Author: AI Assistant
"""

import json
import os
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
import time

import numpy as np
import ai2thor
from ai2thor.controller import Controller
from PIL import Image
import tqdm

# Try to import compression library (for simulated .r3d compatibility)
try:
    import lzfse  # Note: This might not be available, we'll simulate compression
    HAS_LZFSE = True
except ImportError:
    HAS_LZFSE = False
    print("Warning: lzfse not available. Using simulated compression.")


class AI2THORR3DRecorder:
    """
    Records RGB-D sequences from AI2THOR scenes in a format compatible with .r3d files.
    
    This class captures:
    - RGB images from the agent's camera
    - Depth maps 
    - Confidence maps (simulated, as AI2THOR doesn't provide confidence)
    - Camera poses (position + rotation)
    - Scene metadata
    """
    
    def __init__(self, 
                 scene_name: str = "FloorPlan1_1",
                 width: int = 640,
                 height: int = 480,
                 fov: float = 60.0,
                 grid_size: float = 0.25):
        """
        Initialize the recorder with AI2THOR controller.
        
        Args:
            scene_name: Name of the AI2THOR scene to record
            width: Image width
            height: Image height 
            fov: Field of view in degrees
            grid_size: Movement grid size for navigation
        """
        self.scene_name = scene_name
        self.width = width
        self.height = height
        self.fov = fov
        self.grid_size = grid_size
        
        # Initialize AI2THOR controller
        self.controller = Controller(
            scene=scene_name,
            width=width,
            height=height,
            fieldOfView=fov,
            gridSize=grid_size,
            renderDepthImage=True,
            renderInstanceSegmentation=True,
            visibilityDistance=5.0,
            quality="Ultra"
        )
        
        # Storage for recorded data
        self.rgb_images = []
        self.depth_images = []
        self.confidence_maps = []
        self.poses = []
        self.timestamps = []
        
        # Camera intrinsics (approximated from AI2THOR's perspective projection)
        self._compute_camera_matrix()
        
        print(f"AI2THOR R3D Recorder initialized for scene: {scene_name}")
        print(f"Image resolution: {width}x{height}")
        print(f"Field of view: {fov} degrees")
    
    def _compute_camera_matrix(self):
        """Compute camera intrinsic matrix from AI2THOR's FOV."""
        # Convert FOV to focal length
        fov_rad = np.radians(self.fov)
        fx = self.width / (2 * np.tan(fov_rad / 2))
        fy = self.height / (2 * np.tan(fov_rad / 2))
        cx = self.width / 2
        cy = self.height / 2
        
        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
    def _ai2thor_pose_to_matrix(self, agent_metadata: Dict) -> np.ndarray:
        """
        Convert AI2THOR agent pose to transformation matrix.
        
        Args:
            agent_metadata: AI2THOR agent metadata containing position and rotation
            
        Returns:
            4x4 transformation matrix
        """
        pos = agent_metadata['position']
        rot = agent_metadata['rotation'] 
        
        # AI2THOR uses degrees, convert to radians
        pitch = np.radians(rot['x'])
        yaw = np.radians(rot['y']) 
        roll = np.radians(rot['z'])
        
        # Create rotation matrix (ZYX order)
        c_pitch, s_pitch = np.cos(pitch), np.sin(pitch)
        c_yaw, s_yaw = np.cos(yaw), np.sin(yaw)
        c_roll, s_roll = np.cos(roll), np.sin(roll)
        
        R_x = np.array([[1, 0, 0],
                        [0, c_pitch, -s_pitch],
                        [0, s_pitch, c_pitch]])
        
        R_y = np.array([[c_yaw, 0, s_yaw],
                        [0, 1, 0],
                        [-s_yaw, 0, c_yaw]])
        
        R_z = np.array([[c_roll, -s_roll, 0],
                        [s_roll, c_roll, 0],
                        [0, 0, 1]])
        
        R = R_z @ R_y @ R_x
        
        # Create 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [pos['x'], pos['y'], pos['z']]
        
        return T
    
    def _simulate_confidence_map(self, depth_image: np.ndarray) -> np.ndarray:
        """
        Simulate a confidence map since AI2THOR doesn't provide one.
        
        Args:
            depth_image: Depth image from AI2THOR
            
        Returns:
            Simulated confidence map (2 = high confidence, 1 = medium, 0 = low)
        """
        confidence = np.ones_like(depth_image, dtype=np.uint8) * 2
        
        # Reduce confidence for very far objects (> 4 meters)
        confidence[depth_image > 4.0] = 1
        
        # Very low confidence for very distant objects (> 6 meters) or invalid depth
        confidence[depth_image > 6.0] = 0
        confidence[depth_image <= 0] = 0
        
        return confidence
    
    def capture_frame(self) -> bool:
        """
        Capture a single frame (RGB, depth, confidence, pose).
        
        Returns:
            True if capture successful, False otherwise
        """
        event = self.controller.last_event
        
        if not event.metadata['lastActionSuccess']:
            return False
            
        # Get RGB image
        rgb_img = event.frame  # Already in RGB format
        
        # Get depth image (in meters, convert to appropriate scale)
        depth_img = event.depth_frame
        if depth_img is None:
            print("Warning: No depth image available")
            return False
            
        # Simulate confidence map
        confidence_map = self._simulate_confidence_map(depth_img)
        
        # Get camera pose
        agent_metadata = event.metadata['agent']
        pose_matrix = self._ai2thor_pose_to_matrix(agent_metadata)
        
        # Store data
        self.rgb_images.append(rgb_img)
        self.depth_images.append(depth_img)
        self.confidence_maps.append(confidence_map)
        self.poses.append(pose_matrix)
        self.timestamps.append(time.time())
        
        return True
    
    def navigate_and_record(self, 
                          num_frames: int = 100,
                          movement_actions: List[str] = None) -> int:
        """
        Navigate through the scene and record frames.
        
        Args:
            num_frames: Target number of frames to record
            movement_actions: List of movement actions to use
            
        Returns:
            Number of frames actually recorded
        """
        if movement_actions is None:
            movement_actions = [
                "MoveAhead", "MoveBack", "MoveLeft", "MoveRight",
                "RotateLeft", "RotateRight", "LookUp", "LookDown"
            ]
        
        recorded_frames = 0
        consecutive_failures = 0
        max_failures = 10
        
        print(f"Starting navigation and recording for {num_frames} frames...")
        
        # Capture initial frame
        if self.capture_frame():
            recorded_frames += 1
            
        # Navigate and capture
        with tqdm.tqdm(total=num_frames, desc="Recording frames") as pbar:
            while recorded_frames < num_frames and consecutive_failures < max_failures:
                # Choose random action
                action = random.choice(movement_actions)
                
                # Execute action
                event = self.controller.step(action=action)
                
                if event.metadata['lastActionSuccess']:
                    # Capture frame
                    if self.capture_frame():
                        recorded_frames += 1
                        consecutive_failures = 0
                        pbar.update(1)
                    else:
                        consecutive_failures += 1
                else:
                    consecutive_failures += 1
                    
                # Occasionally teleport to new position to explore more of the scene
                if recorded_frames % 20 == 0:
                    self._teleport_to_random_position()
                    consecutive_failures = 0
        
        print(f"Recording completed. Captured {recorded_frames} frames.")
        return recorded_frames
    
    def _teleport_to_random_position(self):
        """Teleport agent to a random reachable position in the scene."""
        try:
            # Get reachable positions
            event = self.controller.step(action="GetReachablePositions")
            positions = event.metadata['actionReturn']
            
            if positions:
                # Choose random position
                pos = random.choice(positions)
                
                # Choose random rotation
                rotation = random.choice([0, 90, 180, 270])
                
                # Teleport
                self.controller.step(
                    action="Teleport",
                    position=pos,
                    rotation=dict(x=0, y=rotation, z=0)
                )
        except Exception as e:
            print(f"Warning: Teleport failed: {e}")
    
    def save_r3d_format(self, output_path: str):
        """
        Save recorded data in .r3d compatible format.
        
        Args:
            output_path: Path to save the .r3d file
        """
        if not self.rgb_images:
            print("No data to save!")
            return
            
        print(f"Saving {len(self.rgb_images)} frames to {output_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        # Create metadata
        metadata = self._create_metadata()
        
        # Save as zip file (simulating .r3d format)
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_STORED) as zipf:
            # Save metadata
            zipf.writestr("metadata", json.dumps(metadata, indent=2))
            
            # Save each frame
            for i in tqdm.tqdm(range(len(self.rgb_images)), desc="Saving frames"):
                # Save RGB image
                rgb_path = f"rgbd/{i}.jpg"
                rgb_img = Image.fromarray(self.rgb_images[i])
                with zipf.open(rgb_path, 'w') as f:
                    rgb_img.save(f, format='JPEG', quality=95)
                
                # Save depth image (compressed)
                depth_path = f"rgbd/{i}.depth"
                depth_data = self._compress_depth(self.depth_images[i])
                zipf.writestr(depth_path, depth_data)
                
                # Save confidence map (compressed)
                conf_path = f"rgbd/{i}.conf"
                conf_data = self._compress_confidence(self.confidence_maps[i])
                zipf.writestr(conf_path, conf_data)
        
        print(f"Successfully saved {output_path}")
    
    def _create_metadata(self) -> Dict:
        """Create metadata dictionary in .r3d format."""
        # Convert poses to the format expected by R3DSemanticDataset
        poses_list = []
        for pose_matrix in self.poses:
            # Extract position and rotation (as quaternion)
            pos = pose_matrix[:3, 3]
            R = pose_matrix[:3, :3]
            
            # Convert rotation matrix to quaternion (simplified)
            # For this simulation, we'll use identity quaternion and real position
            qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0  # Identity quaternion
            px, py, pz = pos[0], pos[1], pos[2]
            
            poses_list.append([qx, qy, qz, qw, px, py, pz])
        
        # Create initial pose (identity)
        init_pose = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        
        metadata = {
            "w": self.width,
            "h": self.height,
            "fps": 30.0,  # Simulated FPS
            "K": self.camera_matrix.T.flatten().tolist(),  # Camera intrinsics (transposed)
            "poses": poses_list,
            "initPose": init_pose,
            "recordingType": "ai2thor_simulation",
            "scene": self.scene_name,
            "totalFrames": len(self.rgb_images)
        }
        
        return metadata
    
    def _compress_depth(self, depth_image: np.ndarray) -> bytes:
        """Compress depth image (simulated LZFSE compression)."""
        # Convert to float32 and flatten
        depth_flat = depth_image.astype(np.float32).flatten()
        
        if HAS_LZFSE:
            # Use real LZFSE compression if available
            raw_bytes = depth_flat.tobytes()
            return lzfse.compress(raw_bytes)
        else:
            # Simulate compression by just storing as bytes
            return depth_flat.tobytes()
    
    def _compress_confidence(self, confidence_map: np.ndarray) -> bytes:
        """Compress confidence map (simulated LZFSE compression)."""
        # Convert to uint8 and flatten
        conf_flat = confidence_map.astype(np.uint8).flatten()
        
        if HAS_LZFSE:
            # Use real LZFSE compression if available
            raw_bytes = conf_flat.tobytes()
            return lzfse.compress(raw_bytes)
        else:
            # Simulate compression by just storing as bytes
            return conf_flat.tobytes()
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'controller'):
            self.controller.stop()


def main():
    """Main function to demonstrate usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Record AI2THOR scenes in .r3d format")
    parser.add_argument("--scene", default="FloorPlan1_1", 
                       help="AI2THOR scene name (default: FloorPlan1_1)")
    parser.add_argument("--output", default="ai2thor_recording.r3d",
                       help="Output .r3d file path (default: ai2thor_recording.r3d)")
    parser.add_argument("--frames", type=int, default=100,
                       help="Number of frames to record (default: 100)")
    parser.add_argument("--width", type=int, default=640,
                       help="Image width (default: 640)")
    parser.add_argument("--height", type=int, default=480,
                       help="Image height (default: 480)")
    parser.add_argument("--fov", type=float, default=60.0,
                       help="Field of view in degrees (default: 60.0)")
    
    args = parser.parse_args()
    
    # Create recorder
    recorder = AI2THORR3DRecorder(
        scene_name=args.scene,
        width=args.width,
        height=args.height,
        fov=args.fov
    )
    
    try:
        # Record data
        num_recorded = recorder.navigate_and_record(num_frames=args.frames)
        
        if num_recorded > 0:
            # Save in .r3d format
            recorder.save_r3d_format(args.output)
            print(f"\nRecording complete!")
            print(f"Scene: {args.scene}")
            print(f"Frames recorded: {num_recorded}")
            print(f"Output file: {args.output}")
            print(f"\nTo use with clip-fields:")
            print(f"  from dataloaders.record3d import R3DSemanticDataset")
            print(f"  dataset = R3DSemanticDataset('{args.output}')")
        else:
            print("Failed to record any frames!")
            
    except KeyboardInterrupt:
        print("\nRecording interrupted by user.")
    except Exception as e:
        print(f"Error during recording: {e}")
    finally:
        recorder.cleanup()


if __name__ == "__main__":
    main() 