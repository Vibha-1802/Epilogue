import torch
import torch.nn as nn
import os
import sys

# Ensure the PointNet++ repository is in the path
repo_path = os.path.abspath('Pointnet_Pointnet2_pytorch')
if repo_path not in sys.path:
    sys.path.append(repo_path)
import models.pointnet2_sem_seg as pointnet2_sem_seg
from models.pointnet2_utils import PointNetSetAbstraction

def get_outdoor_model(num_classes=19, input_channels=1):
    """
    Creates a PointNet++ semantic segmentation model tailored for outdoor LiDAR data.
    
    Args:
        num_classes: Number of semantic classes to output (default 19 for SemanticKITTI)
        input_channels: Number of additional feature channels besides XYZ. 
                        For LiDAR (XYZI), input_channels is 1 (Intensity).
    """
    # Initialize the base model (the num_classes passed here will set up the initial final layer)
    model = pointnet2_sem_seg.get_model(num_classes)
    
    # 1. Modify the first Set Abstraction layer (`sa1`)
    # The original was: PointNetSetAbstraction(1024, 0.1, 32, 9 + 3, [32, 32, 64], False)
    # We change it to accept input_channels + 3 (XYZ).
    model.sa1 = PointNetSetAbstraction(1024, 0.1, 32, input_channels + 3, [32, 32, 64], False)
    
    # Note: The output layer (`model.conv2`) is already correctly sized because we passed 
    # num_classes to get_model() above, but we explicitly reinitialize it just to be safe
    # and ensure its weights are totally reset for the new task.
    model.conv2 = nn.Conv1d(128, num_classes, 1)
    
    return model

def load_pretrained_weights(model, checkpoint_path):
    """
    Loads pre-trained weights from the S3DIS indoor model into our modified outdoor model.
    It skips the first (sa1) and last (conv2) layers since their shapes have changed.
    """
    # Load the checkpoint (weights_only=False is needed for older checkpoints in PyTorch 2.6+)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    pretrained_dict = checkpoint['model_state_dict']
    
    # Get the current model's dictionary
    model_dict = model.state_dict()
    
    # Filter out unnecessary keys (keys that don't match in shape)
    filtered_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            filtered_dict[k] = v
        else:
            print(f"Skipping weight transfer for: {k}")
            
    # Overwrite entries in the existing state dict
    model_dict.update(filtered_dict)
    
    # Load the new state dict
    model.load_state_dict(model_dict)
    print("Pre-trained weights loaded successfully (skipped incompatible layers).")
    
    return model

# Test code
if __name__ == '__main__':
    # Test if model instantiates and accepts [1, 4, 4096] tensor
    dummy_model = get_outdoor_model(num_classes=19, input_channels=1)
    dummy_input = torch.rand(1, 4, 4096) # XYZI -> 4 channels total
    
    with torch.no_grad():
        out, _ = dummy_model(dummy_input)
        
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (1, 4096, 19), "Output shape mismatch!"
    print("Outdoor PointNet++ wrapper works correctly!")
