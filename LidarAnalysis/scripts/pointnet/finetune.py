import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from semantickitti_dataset import SemanticKITTIDataset
from outdoor_pointnet import get_outdoor_model, load_pretrained_weights

def main():
    # --- Configuration ---
    DATASET_PATH = 'dataset/sequences'
    CHECKPOINT_PATH = 'Pointnet_Pointnet2_pytorch/log/sem_seg/pointnet2_sem_seg/checkpoints/best_model.pth'
    
    BATCH_SIZE = 4
    NUM_POINTS = 4096
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 19
    INPUT_CHANNELS = 1 # Intensity
    
    # Check if dataset path exists to avoid crashing immediately
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: SemanticKITTI dataset not found at {DATASET_PATH}.")
        print("Please ensure the data is downloaded and extracted properly.")
        return

    # --- 1. Dataset & DataLoader ---
    print("Initializing datasets...")
    # Using seq 00 for training, seq 08 for validation as is standard for SemanticKITTI
    train_dataset = SemanticKITTIDataset(DATASET_PATH, sequences=['00'], num_points=NUM_POINTS, split='train')
    val_dataset = SemanticKITTIDataset(DATASET_PATH, sequences=['08'], num_points=NUM_POINTS, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=False)
    
    # --- 2. Model & Transfer Learning ---
    print("Initializing Outdoor PointNet++...")
    model = get_outdoor_model(num_classes=NUM_CLASSES, input_channels=INPUT_CHANNELS)
    
    if os.path.exists(CHECKPOINT_PATH):
        print("Loading pre-trained S3DIS weights...")
        model = load_pretrained_weights(model, CHECKPOINT_PATH)
    else:
        print(f"WARNING: Pre-trained weights not found at {CHECKPOINT_PATH}. Training from scratch!")
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # --- 3. Loss & Optimizer ---
    # SemanticKITTI is highly imbalanced. Ideally, we calculate these weights from the dataset.
    # For now, we use uniform weights, but you can plug in inverse frequency weights here.
    class_weights = torch.ones(NUM_CLASSES).to(device) 
    class_weights[0] = 0.0 # Ignore class 0 (unlabeled) in the loss calculation
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # We use Adam optimizer with a lower learning rate to preserve transferred weights
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # --- 4. Training Loop ---
    print("Starting Training Loop...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        # tqdm progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for batch_features, batch_labels in pbar:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions, trans_feat = model(batch_features)
            
            # PointNet++ outputs (Batch, Points, Classes)
            # PyTorch CrossEntropy expects (Batch, Classes, Points)
            predictions = predictions.transpose(1, 2)
            
            # Calculate loss
            loss = criterion(predictions, batch_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Average Train Loss: {avg_train_loss:.4f}")
        
        # Save checkpoint periodically
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(model.state_dict(), f'checkpoints/semantickitti_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    main()
