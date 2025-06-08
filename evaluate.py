import torch
import torch.nn.functional as F
import clip
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
import tqdm
import numpy as np
from grid_hash_model import GridCLIPModel
from dataloaders.real_dataset import DeticDenseLabelledDataset
import hydra
from omegaconf import OmegaConf
import os
import open3d as o3d
import matplotlib.pyplot as plt

def load_models_and_data(model_path, dataset_path, cfg):
    """Load the trained model and dataset using the same configuration as training."""
    # Load CLIP and SentenceTransformer models with same versions as training
    clip_model, _ = clip.load(cfg.web_models.clip, device=cfg.device)
    sentence_model = SentenceTransformer(cfg.web_models.sentence).to(cfg.device)
    
    # Load dataset
    training_data = torch.load(dataset_path)
    max_coords, _ = training_data._label_xyz.max(dim=0)
    min_coords, _ = training_data._label_xyz.min(dim=0)
    
    # Initialize model with same parameters as training
    label_model = GridCLIPModel(
        image_rep_size=training_data[0]["clip_image_vector"].shape[-1],
        text_rep_size=training_data[0]["clip_vector"].shape[-1],
        mlp_depth=cfg.mlp_depth,
        mlp_width=cfg.mlp_width,
        log2_hashmap_size=cfg.log2_hashmap_size,
        num_levels=cfg.num_grid_levels,
        level_dim=cfg.level_dim,
        per_level_scale=cfg.per_level_scale,
        max_coords=max_coords,
        min_coords=min_coords,
    ).to(cfg.device)
    
    model_weights = torch.load(model_path, map_location=cfg.device)
    label_model.load_state_dict(model_weights["model"])
    
    return clip_model, sentence_model, label_model, training_data

def calculate_embeddings(query, clip_model, sentence_model, device):
    """Calculate CLIP and SentenceTransformer embeddings for a query."""
    with torch.no_grad():
        clip_tokens = clip_model.encode_text(clip.tokenize([query]).to(device)).float()
        clip_tokens = F.normalize(clip_tokens, p=2, dim=-1)
        st_tokens = torch.from_numpy(sentence_model.encode([query])).to(device)
        st_tokens = F.normalize(st_tokens, p=2, dim=-1)
    return clip_tokens, st_tokens

def find_best_point(query, label_model, training_data, clip_model, sentence_model, cfg, visual=False):
    """Find the point that best matches the query."""
    # Calculate embeddings for the query
    clip_tokens, st_tokens = calculate_embeddings(query, clip_model, sentence_model, cfg.device)
    
    # Set weights for visual vs semantic alignment using same ratios as training
    vision_weight = cfg.image_to_label_loss_ratio
    text_weight = cfg.label_to_image_loss_ratio
    
    if visual:
        vision_weight *= 10.0
        text_weight *= 1.0
    else:
        vision_weight *= 1.0
        text_weight *= 10.0
    
    # Create dataloader with same batch size as training
    points_dataloader = DataLoader(
        training_data._label_xyz, 
        batch_size=cfg.batch_size, 
        num_workers=cfg.num_workers
    )
    
    best_alignment = float('-inf')
    best_point = None
    
    with torch.no_grad():
        for points in tqdm.tqdm(points_dataloader, desc="Finding best point"):
            # Get model predictions
            predicted_label_latents, predicted_image_latents = label_model(points.to(cfg.device))
            
            # Normalize predictions
            data_text_tokens = F.normalize(predicted_label_latents, p=2, dim=-1)
            data_visual_tokens = F.normalize(predicted_image_latents, p=2, dim=-1)
            
            # Calculate alignments
            text_alignment = data_text_tokens @ st_tokens.T
            visual_alignment = data_visual_tokens @ clip_tokens.T
            
            # Combine alignments with weights
            total_alignment = (text_weight * text_alignment + vision_weight * visual_alignment) / (text_weight + vision_weight)
            
            # Find best alignment in this batch
            batch_best_alignment = total_alignment.max().item()
            batch_best_idx = total_alignment.argmax().item()
            
            if batch_best_alignment > best_alignment:
                best_alignment = batch_best_alignment
                best_point = points[batch_best_idx].cpu().numpy()
    
    return best_point, best_alignment

def visualize_point_cloud(training_data, best_point, query):
    """Save the point cloud and best match as .ply files only (no matplotlib visualization)."""
    # Get point cloud data
    points = training_data._label_xyz.cpu().numpy()
    colors = training_data._label_rgb.cpu().numpy()
    
    # Normalize colors to [0, 1] range
    colors_norm = colors / 255.0
    
    # --- Save the full point cloud as a .ply file ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors_norm)
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    ply_filename = os.path.join(output_dir, "scene_pointcloud.ply")
    o3d.io.write_point_cloud(ply_filename, pcd)
    print(f"Point cloud saved to: {ply_filename}")
    
    # --- Save the best match point as a separate .ply file ---
    best_pcd = o3d.geometry.PointCloud()
    best_pcd.points = o3d.utility.Vector3dVector([best_point])
    best_pcd.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]])  # Red
    best_ply_filename = os.path.join(output_dir, "best_match_point.ply")
    o3d.io.write_point_cloud(best_ply_filename, best_pcd)
    print(f"Best match point saved to: {best_ply_filename}")
    
    print(f"Best match coordinates: X={best_point[0]:.2f}, Y={best_point[1]:.2f}, Z={best_point[2]:.2f}")

@hydra.main(version_base="1.2", config_path="configs", config_name="train.yaml")
def main(cfg):
    # Get the original working directory
    original_cwd = hydra.utils.get_original_cwd()
    
    # Set up paths
    model_path = os.path.join(original_cwd, "clip_implicit_model/implicit_scene_label_model_latest.pt")
    dataset_path = os.path.join(original_cwd, "detic_labeled_dataset.pt")
    
    # Load models and data
    clip_model, sentence_model, label_model, training_data = load_models_and_data(
        model_path, dataset_path, cfg
    )
    
    # Find best point
    best_point, alignment = find_best_point(
        cfg.query, 
        label_model, 
        training_data, 
        clip_model, 
        sentence_model,
        cfg,
        visual=cfg.get("visual", False)
    )
    
    print(f"\nQuery: {cfg.query}")
    print(f"Best point coordinates: {best_point}")
    print(f"Alignment score: {alignment:.4f}")
    
    # Visualize the result
    visualize_point_cloud(training_data, best_point, cfg.query)

if __name__ == "__main__":
    main() 