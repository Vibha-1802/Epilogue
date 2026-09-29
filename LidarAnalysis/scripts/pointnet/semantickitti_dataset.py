import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class SemanticKITTIDataset(Dataset):
    def __init__(self, data_path, sequences=['00'], num_points=4096, split='train'):
        """
        Args:
            data_path: Path to dataset/sequences/
            sequences: List of sequence strings (e.g., ['00', '01'])
            num_points: Number of points to sample per frame
            split: 'train' or 'val'
        """
        self.data_path = data_path
        self.num_points = num_points
        self.split = split
        
        # Standard SemanticKITTI learning map (maps 20+ classes down to 19 valid ones)
        # Anything mapped to 0 is 'unlabeled' or 'ignored'
        self.learning_map = {
            0 : 0,     # "unlabeled", and others ignored
            1 : 0,     # "outlier" mapped to "unlabeled"
            10: 1,     # car
            11: 2,     # bicycle
            13: 5,     # bus mapped to other-vehicle
            15: 3,     # motorcycle
            16: 5,     # on-rails mapped to other-vehicle
            18: 4,     # truck
            20: 5,     # other-vehicle
            30: 6,     # person
            31: 7,     # bicyclist
            32: 8,     # motorcyclist
            40: 9,     # road
            44: 10,    # parking
            48: 11,    # sidewalk
            49: 12,    # other-ground
            50: 13,    # building
            51: 14,    # fence
            52: 0,     # other-structure mapped to unlabeled
            60: 9,     # lane-marking to road
            70: 15,    # vegetation
            71: 16,    # trunk
            72: 17,    # terrain
            80: 18,    # pole
            81: 19,    # traffic-sign
            99: 0,     # other-object to unlabeled
            252: 1,    # moving-car to car
            253: 7,    # moving-bicyclist to bicyclist
            254: 6,    # moving-person to person
            255: 8,    # moving-motorcyclist to motorcyclist
            256: 5,    # moving-on-rails mapped to other-vehicle
            257: 5,    # moving-bus mapped to other-vehicle
            258: 4,    # moving-truck to truck
            259: 5,    # moving-other-vehicle to other-vehicle
        }
        
        self.learning_map_lut = np.zeros((260,), dtype=np.int32)
        for k, v in self.learning_map.items():
            self.learning_map_lut[k] = v
            
        self.scan_files = []
        self.label_files = []
        
        # Collect all bin and label files for the specified sequences
        for seq in sequences:
            scan_path = os.path.join(data_path, seq, 'velodyne')
            label_path = os.path.join(data_path, seq, 'labels')
            
            scans = sorted(glob.glob(os.path.join(scan_path, '*.bin')))
            labels = sorted(glob.glob(os.path.join(label_path, '*.label')))
            
            # Ensure they match
            assert len(scans) == len(labels), f"Mismatch between scans and labels in seq {seq}"
            
            self.scan_files.extend(scans)
            self.label_files.extend(labels)
            
        print(f"Loaded {len(self.scan_files)} frames for {self.split} split.")

    def __len__(self):
        return len(self.scan_files)

    def __getitem__(self, idx):
        scan_file = self.scan_files[idx]
        label_file = self.label_files[idx]
        
        # 1. Read scan (X, Y, Z, Intensity)
        scan = np.fromfile(scan_file, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        
        # 2. Read label (uint32)
        label = np.fromfile(label_file, dtype=np.uint32)
        
        # 3. Extract semantic class (lower 16 bits) and remap
        sem_label = label & 0xFFFF
        sem_label = self.learning_map_lut[sem_label]
        
        # 4. Sample exactly `num_points`
        num_raw = scan.shape[0]
        if num_raw >= self.num_points:
            choice = np.random.choice(num_raw, self.num_points, replace=False)
        else:
            choice = np.random.choice(num_raw, self.num_points, replace=True)
            
        sampled_scan = scan[choice, :]
        sampled_label = sem_label[choice]
        
        # Format for PyTorch and PointNet++
        # PointNet++ usually expects shape (C, N), so we transpose the features
        point_features = torch.tensor(sampled_scan, dtype=torch.float32).transpose(0, 1)
        point_labels = torch.tensor(sampled_label, dtype=torch.long)
        
        return point_features, point_labels

# Test code if run directly
if __name__ == '__main__':
    # You would point this to your unzipped dataset folder
    dummy_path = 'dataset/sequences'
    if os.path.exists(dummy_path):
        dataset = SemanticKITTIDataset(dummy_path, sequences=['00'])
        feats, labels = dataset[0]
        print("Features shape:", feats.shape)  # Should be (4, 4096)
        print("Labels shape:", labels.shape)   # Should be (4096,)
