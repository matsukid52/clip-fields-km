#!/usr/bin/env python3
"""
Interactive AI2-THOR Scene Recorder for clip-fields

This script launches an interactive AI2-THOR session allowing a user to navigate
a scene and record RGB-D data. The recorded data is packaged into an `.r3d` 
file, which is a zip archive compatible with the clip-fields training pipeline.
"""

import os
import json
import time
import zipfile
import argparse
from pathlib import Path
import shutil
import datetime

import numpy as np
import pygame
from PIL import Image
import lzfse
from pyquaternion import Quaternion
from tqdm import tqdm
import ai2thor.controller
import open3d as o3d
# from quaternion import as_rotation_matrix, quaternion as np_quaternion

class InteractiveRecorder:
    def __init__(self, scene_name="FloorPlan1", width=640, height=480, fov=90, frame_rate=30, mouse_sensitivity=0.1):
        self.scene_name = scene_name
        self.width = int(width)
        self.height = int(height)
        self.fov = fov
        self.frame_rate = frame_rate
        self.frame_duration = 1.0 / frame_rate
        self.mouse_sensitivity = mouse_sensitivity

        self.is_recording = False
        self.recorded_data = []
        self.init_pose = None
        
        self.controller = ai2thor.controller.Controller(
            agentMode="locobot",
            visibilityDistance=1.5,
            scene=self.scene_name,
            # Render RGB in high-res, but depth in low-res, as clip-fields will upscale it
            width=self.width,
            height=self.height,
            renderDepthImage=True,
            renderInstanceSegmentation=False,
        )
        
        # Get low-res depth dimensions from the controller, which is what clip-fields expects
        self.depth_width = self.controller.last_event.depth_frame.shape[1]
        self.depth_height = self.controller.last_event.depth_frame.shape[0]

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("AI2-THOR Interactive Recorder")
        self.font = pygame.font.SysFont("Arial", 24)

        # For mouse control
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

    def run(self):
        running = True
        last_frame_time = 0
        
        while running:
            current_time = time.time()
            if current_time - last_frame_time < self.frame_duration:
                time.sleep(self.frame_duration - (current_time - last_frame_time))
            last_frame_time = time.time()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self.handle_key_press(event.key)
                elif event.type == pygame.MOUSEMOTION:
                    self.handle_mouse_motion(event)

            self.update_display()
            
        self.quit()

    def handle_key_press(self, key):
        action = None
        if key == pygame.K_w:
            action = "MoveAhead"
        elif key == pygame.K_s:
            action = "MoveBack"
        elif key == pygame.K_a:
            action = "MoveLeft"
        elif key == pygame.K_d:
            action = "MoveRight"
        elif key == pygame.K_r:
            self.toggle_recording()
        elif key == pygame.K_q or key == pygame.K_ESCAPE:
            return False

        if action:
            self.last_event = self.controller.step(action)
            if self.is_recording:
                self.capture_frame()
        return True

    def handle_mouse_motion(self, event):
        motion_happened = False
        dx, dy = event.rel
        
        # Horizontal rotation (yaw)
        if dx != 0:
            rotation_degrees = dx * self.mouse_sensitivity
            self.last_event = self.controller.step(
                action="RotateRight",
                degrees=rotation_degrees
            )
            motion_happened = True

        # Vertical rotation (pitch)
        if dy != 0:
            look_degrees = dy * self.mouse_sensitivity
            self.last_event = self.controller.step(
                action="LookUp",
                degrees=-look_degrees # Negative for natural up/down
            )
            motion_happened = True
        
        if self.is_recording and motion_happened:
            self.capture_frame()

    def toggle_recording(self):
        self.is_recording = not self.is_recording
        if self.is_recording:
            print("--- Started Recording ---")
            self.recorded_data = []
            self.last_event = self.controller.step("Pass") # Ensure we have a valid event
            self.init_pose = self._get_pose_from_event(self.last_event)
            self.capture_frame()
        else:
            print("--- Stopped Recording ---")
            if self.recorded_data:
                self.save_r3d()

    def capture_frame(self):
        event = self.last_event
        if not event.metadata["lastActionSuccess"]:
            return

        pose = self._get_pose_from_event(event)
        
        frame_data = {
            "rgb": event.frame,
            "depth": event.depth_frame,
            "pose": pose,
        }
        self.recorded_data.append(frame_data)

    def _get_pose_from_event(self, event):
        # This function now returns the raw pose that R3DSemanticDataset expects
        agent_meta = event.metadata['agent']
        pos = agent_meta['position']
        
        # Get body yaw and camera pitch
        yaw = agent_meta['rotation']['y']
        pitch = agent_meta['cameraHorizon']

        # R3DSemanticDataset uses pyquaternion, so we match it.
        # It expects a specific coordinate system transformation.
        # We will store the AI2-THOR pose and transform it later if needed,
        # but R3DSemanticDataset does its own transform.
        # Let's match the final pose format from R3DSemanticDataset.
        
        # Transformation from AI2-THOR coords to Record3D/clip-fields coords
        # AI2-THOR: +X right, +Y up, +Z forward
        # Record3D: +X right, +Y down, +Z forward
        px, py, pz = pos['x'], -pos['y'], pos['z']

        # Rotation
        # AI2-THOR rot: pitch (around x), yaw (around y)
        # We create quaternions and then apply the coordinate system change.
        q_pitch = Quaternion(axis=[1, 0, 0], degrees=pitch)
        q_yaw = Quaternion(axis=[0, 1, 0], degrees=yaw)
        
        # This is the key transformation to match Record3D's camera orientation
        q_coord_system_change = Quaternion(axis=[1, 0, 0], degrees=180)
        
        q_final = q_coord_system_change * q_pitch * q_yaw
        
        # Return in [qx, qy, qz, qw, px, py, pz] format
        qw, qx, qy, qz = q_final.elements
        return [qx, qy, qz, qw, px, py, pz]

    def get_intrinsics(self):
        # This must match the RGB dimensions, as depth will be upscaled to it
        w, h = self.width, self.height
        fov = self.fov
        
        fx = (w / 2.0) / np.tan(np.deg2rad(fov / 2.0))
        fy = fx # Assuming square pixels for simplicity, consistent with clip-fields
        cx = w / 2.0
        cy = h / 2.0
        
        # R3DSemanticDataset expects K.T, so we provide it transposed
        return [fx, 0, cx, 0, fy, cy, 0, 0, 1]

    def save_r3d(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{self.scene_name}_{timestamp}.r3d"
        temp_dir = Path(f"temp_r3d_{timestamp}")
        rgbd_dir = temp_dir / "rgbd"
        rgbd_dir.mkdir(parents=True, exist_ok=True)

        print(f"Saving {len(self.recorded_data)} frames to {output_filename}...")

        # Prepare metadata. This now reflects the true data being saved.
        poses = [frame["pose"] for frame in self.recorded_data]
        metadata = {
            "w": self.width,
            "h": self.height,
            "depth_w": self.depth_width, # Store original depth dimensions
            "depth_h": self.depth_height,
            "fps": self.frame_rate,
            "K": self.get_intrinsics(),
            "poses": poses,
            "initPose": self.init_pose
        }
        with open(temp_dir / "metadata", "w") as f:
            json.dump(metadata, f)
            
        # Process and save frames
        for i, frame_data in tqdm(enumerate(self.recorded_data), total=len(self.recorded_data), desc="Processing frames"):
            # Save RGB (High-Res)
            rgb_img = Image.fromarray(frame_data["rgb"])
            rgb_img.save(rgbd_dir / f"{i}.jpg")
            
            # Save Depth (Low-Res, Float32)
            depth_map = frame_data["depth"].astype(np.float32)
            compressed_depth = lzfse.compress(depth_map.tobytes())
            with open(rgbd_dir / f"{i}.depth", "wb") as f:
                f.write(compressed_depth)

            # Save Confidence map (matches low-res depth)
            conf_map = (np.ones_like(depth_map, dtype=np.uint8) * 2) # All points are confident
            compressed_conf = lzfse.compress(conf_map.tobytes())
            with open(rgbd_dir / f"{i}.conf", "wb") as f:
                f.write(compressed_conf)
                
        # Create zip file
        with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in temp_dir.rglob('*'):
                zf.write(file_path, file_path.relative_to(temp_dir))
                
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"Successfully created {output_filename}")

    def update_display(self):
        frame = self.controller.last_event.frame
        pygame_frame = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
        self.screen.blit(pygame_frame, (0, 0))
        
        status_text = "RECORDING" if self.is_recording else "Press 'R' to record"
        color = (255, 0, 0) if self.is_recording else (255, 255, 255)
        text_surface = self.font.render(status_text, True, color)
        self.screen.blit(text_surface, (10, 10))

        pygame.display.flip()

    def quit(self):
        pygame.mouse.set_visible(True)
        pygame.event.set_grab(False)
        self.controller.stop()
        pygame.quit()
        print("Recorder session ended.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI2-THOR Interactive Recorder for clip-fields")
    parser.add_argument("--scene", type=str, default="FloorPlan_Train1_1", help="AI2-THOR scene name to load.")
    parser.add_argument("--width", type=int, default=640, help="Width of the RGB window.")
    parser.add_argument("--height", type=int, default=480, help="Height of the RGB window.")
    parser.add_argument("--fov", type=int, default=90, help="Field of View.")
    parser.add_argument("--fps", type=int, default=30, help="Recording frame rate.")
    parser.add_argument("--mouse_sensitivity", type=float, default=0.2, help="Mouse sensitivity for rotation.")
    
    args = parser.parse_args()
    
    recorder = InteractiveRecorder(
        scene_name=args.scene,
        width=args.width,
        height=args.height,
        fov=args.fov,
        frame_rate=args.fps,
        mouse_sensitivity=args.mouse_sensitivity
    )
    recorder.run() 