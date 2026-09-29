import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from semantickitti_dataset import SemanticKITTIDataset
from outdoor_pointnet import get_outdoor_model

def get_semantickitti_colors():
    """
    Returns a dictionary mapping the 0-19 evaluation class IDs to RGB colors.
    """
    return {
        0: [0, 0, 0],         # unlabeled / ignored (Black)
        1: [100, 150, 245],   # car (Light Blue)
        2: [100, 230, 245],   # bicycle (Cyan)
        3: [30, 60, 150],     # motorcycle (Dark Blue)
        4: [80, 30, 180],     # truck (Purple)
        5: [100, 80, 250],    # other-vehicle (Violet)
        6: [255, 30, 30],     # person (Red)
        7: [255, 40, 200],    # bicyclist (Pink)
        8: [150, 30, 90],     # motorcyclist (Dark Pink)
        9: [255, 0, 255],     # road (Magenta)
        10: [255, 150, 255],  # parking (Light Magenta)
        11: [75, 0, 75],      # sidewalk (Dark Magenta)
        12: [175, 0, 75],     # other-ground (Maroon)
        13: [255, 200, 0],    # building (Orange)
        14: [255, 120, 50],   # fence (Orange-Red)
        15: [0, 175, 0],      # vegetation (Green)
        16: [135, 60, 0],     # trunk (Brown)
        17: [150, 240, 80],   # terrain (Light Green)
        18: [255, 240, 150],  # pole (Yellow)
        19: [255, 0, 0]       # traffic-sign (Red)
    }

def main():
    # --- Config ---
    DATASET_PATH = '/content/dataset_trimmed/sequences'
    # Try epoch 10 first, fallback to epoch 1 if it doesn't exist
    CHECKPOINT_PATH_10 = '/content/drive/MyDrive/checkpoints/semantickitti_epoch_10.pth'
    CHECKPOINT_PATH_1 = '/content/drive/MyDrive/checkpoints/semantickitti_epoch_1.pth'
    
    NUM_CLASSES = 20
    INPUT_CHANNELS = 1
    
    # --- Load Data ---
    print("Loading test frame from SemanticKITTI (Sequence 00)...")
    dataset = SemanticKITTIDataset(DATASET_PATH, sequences=['00'], num_points=4096, split='val')
    
    # Grab the very first frame
    point_features, point_labels = dataset[0]
    
    # Features shape: (4, 4096). We need to add batch dimension -> (1, 4, 4096)
    inputs = point_features.unsqueeze(0)
    labels = point_labels.numpy() # (4096,)
    
    # Extract XYZ for plotting
    # inputs is (1, 4, 4096), meaning [Batch, Channel, Points]
    # Channels are [X, Y, Z, Intensity]
    xyz = inputs[0, :3, :].transpose(0, 1).numpy() # (4096, 3)
    
    # --- Load Model ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_outdoor_model(num_classes=NUM_CLASSES, input_channels=INPUT_CHANNELS)
    
    if os.path.exists(CHECKPOINT_PATH_10):
        print(f"Loading weights from {CHECKPOINT_PATH_10}...")
        model.load_state_dict(torch.load(CHECKPOINT_PATH_10, map_location=device, weights_only=False))
    elif os.path.exists(CHECKPOINT_PATH_1):
        print(f"Loading weights from {CHECKPOINT_PATH_1}...")
        model.load_state_dict(torch.load(CHECKPOINT_PATH_1, map_location=device, weights_only=False))
    else:
        print("ERROR: Could not find epoch 1 or epoch 10 checkpoint in your Google Drive!")
        return
        
    model = model.to(device)
    model.eval()
    
    # --- Run Inference ---
    print("Running inference...")
    inputs = inputs.to(device)
    with torch.no_grad():
        predictions, _ = model(inputs)
    
    # predictions is (1, 4096, 20)
    # Get the argmax across the classes (dim=2)
    pred_labels = torch.argmax(predictions, dim=2).squeeze(0).cpu().numpy()
    
    # --- Visualization ---
    print("Plotting results...")
    color_map = get_semantickitti_colors()
    
    # Map label integers to normalized RGB (0-1) arrays
    gt_colors = np.array([color_map[l] for l in labels]) / 255.0
    pred_colors = np.array([color_map[l] for l in pred_labels]) / 255.0
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Ground Truth Plot
    ax1.scatter(xyz[:, 0], xyz[:, 1], c=gt_colors, s=5, alpha=0.8)
    ax1.set_title("Ground Truth", fontsize=16)
    ax1.axis('equal')
    ax1.set_facecolor('black') # Black background looks better for LiDAR
    
    # Prediction Plot
    ax2.scatter(xyz[:, 0], xyz[:, 1], c=pred_colors, s=5, alpha=0.8)
    ax2.set_title("PointNet++ Prediction", fontsize=16)
    ax2.axis('equal')
    ax2.set_facecolor('black')
    
    plt.tight_layout()
    plt.savefig('semantickitti_test_result.png')
    plt.show()
    print("Done! Saved plot to semantickitti_test_result.png")

if __name__ == '__main__':
    main()
