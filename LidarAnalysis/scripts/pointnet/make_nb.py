import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# End-to-End LiDAR Pipeline: Preprocessing -> PointNet++\n",
    "This notebook reads a LiDAR `.pcd` frame, cleans it, runs Adaptive Spatial Sampling, maps the features to the 9-channel format required by the S3DIS pretrained model, and runs PointNet++ inference."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git\n",
    "!pip install -q pypcd4 pandas matplotlib"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import torch\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from pypcd4 import PointCloud\n",
    "import matplotlib.pyplot as plt\n",
    "import sys\n",
    "\n",
    "sys.path.append('Pointnet_Pointnet2_pytorch/models')\n",
    "import pointnet2_sem_seg\n",
    "\n",
    "print(\"PyTorch:\", torch.__version__)\n",
    "print(\"CUDA:\", torch.cuda.is_available())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Upload your PCD file to the Colab session first!\n",
    "PCD_FILE = \"frame_00006.pcd\" # Change this if your uploaded file has a different name\n",
    "\n",
    "if not os.path.exists(PCD_FILE):\n",
    "    print(f\"\\n⚠️ Please upload {PCD_FILE} to the Colab files pane on the left!\")\n",
    "else:\n",
    "    # Read and clean using pypcd4\n",
    "    pc = PointCloud.from_path(PCD_FILE)\n",
    "    xyz = pc.pc_data[['x', 'y', 'z']]\n",
    "    points = np.vstack([xyz['x'], xyz['y'], xyz['z']]).T\n",
    "    valid_mask = np.isfinite(points).all(axis=1)\n",
    "    valid_points = points[valid_mask]\n",
    "    \n",
    "    if 'intensity' in pc.fields:\n",
    "        intensities = pc.pc_data['intensity'][valid_mask].flatten()\n",
    "    else:\n",
    "        intensities = np.zeros(len(valid_points))\n",
    "        \n",
    "    points_with_intensity = np.hstack((valid_points, intensities.reshape(-1, 1)))\n",
    "    \n",
    "    # Add r, theta\n",
    "    x = points_with_intensity[:, 0]\n",
    "    y = points_with_intensity[:, 1]\n",
    "    r = np.sqrt(x**2 + y**2)\n",
    "    theta = np.arctan2(y, x)\n",
    "    points_XYZIrtheta = np.hstack((points_with_intensity, r.reshape(-1, 1), theta.reshape(-1, 1)))\n",
    "    \n",
    "    print(f\"Extracted {len(points_XYZIrtheta)} valid points from PCD.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def adaptive_sample(points_array):\n",
    "    bands = [(0, 50, 0.05), (50, 85, 0.25), (85, 100, 0.50)]\n",
    "    df = pd.DataFrame(points_array, columns=['x', 'y', 'z', 'intensity', 'r', 'theta'])\n",
    "    df_filtered = df[df['r'] <= 100.0].copy()\n",
    "    \n",
    "    aggregated_records = []\n",
    "    for r_min, r_max, cell_size in bands:\n",
    "        mask = (df_filtered['r'] >= r_min) & (df_filtered['r'] <= r_max) if r_min == 0 else (df_filtered['r'] > r_min) & (df_filtered['r'] <= r_max)\n",
    "        df_band = df_filtered[mask].copy()\n",
    "        if df_band.empty: continue\n",
    "            \n",
    "        df_band['x_idx'] = np.floor(df_band['x'] / cell_size).astype(int)\n",
    "        df_band['y_idx'] = np.floor(df_band['y'] / cell_size).astype(int)\n",
    "        \n",
    "        agg_df = df_band.groupby(['x_idx', 'y_idx']).agg(\n",
    "            representative_x=('x', 'mean'),\n",
    "            representative_y=('y', 'mean'),\n",
    "            representative_z=('z', 'mean'),\n",
    "            representative_intensity=('intensity', 'mean')\n",
    "        ).reset_index()\n",
    "        aggregated_records.append(agg_df)\n",
    "        \n",
    "    return pd.concat(aggregated_records, ignore_index=True) if aggregated_records else pd.DataFrame()\n",
    "\n",
    "if 'points_XYZIrtheta' in locals():\n",
    "    aggregated_data = adaptive_sample(points_XYZIrtheta)\n",
    "    print(f\"Adaptive Sampling reduced points to: {len(aggregated_data)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# PointNet++ preparation: Pad C=4 to C=9 & Format to tensor\n",
    "if 'aggregated_data' in locals():\n",
    "    # PointNet++ needs EXACTLY 4096 points for this architecture configuration\n",
    "    NUM_POINTS = 4096\n",
    "    xyz_intensity = aggregated_data[['representative_x', 'representative_y', 'representative_z', 'representative_intensity']].values\n",
    "    \n",
    "    if len(xyz_intensity) > NUM_POINTS:\n",
    "        indices = np.random.choice(len(xyz_intensity), NUM_POINTS, replace=False)\n",
    "    else:\n",
    "        indices = np.random.choice(len(xyz_intensity), NUM_POINTS, replace=True)\n",
    "    \n",
    "    sampled_points = xyz_intensity[indices]\n",
    "    \n",
    "    xyz = sampled_points[:, :3]\n",
    "    intensity = sampled_points[:, 3:4]\n",
    "    \n",
    "    # Map 4 channels to 9 channels: [X, Y, Z, R, G, B, norm_x, norm_y, norm_z]\n",
    "    # Fake RGB using intensity\n",
    "    rgb = np.hstack([intensity, intensity, intensity]) / 255.0\n",
    "    \n",
    "    # Normalize XYZ\n",
    "    xyz_min, xyz_max = np.min(xyz, axis=0), np.max(xyz, axis=0)\n",
    "    xyz_normalized = (xyz - xyz_min) / (xyz_max - xyz_min + 1e-6)\n",
    "    \n",
    "    # Concatenate to 9 channels\n",
    "    features_9 = np.hstack([xyz, rgb, xyz_normalized])\n",
    "    \n",
    "    # PyTorch wants shape: (Batch, Channels, Points)\n",
    "    tensor_data = torch.tensor(features_9, dtype=torch.float32).transpose(0, 1).unsqueeze(0)\n",
    "    if torch.cuda.is_available(): tensor_data = tensor_data.cuda()\n",
    "        \n",
    "    print(f\"Final Input Tensor Shape: {tensor_data.shape}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load Model and Run Inference\n",
    "if 'tensor_data' in locals():\n",
    "    model = pointnet2_sem_seg.get_model(13) # 13 S3DIS Classes\n",
    "    \n",
    "    checkpoint_path = 'Pointnet_Pointnet2_pytorch/log/sem_seg/pointnet2_sem_seg/checkpoints/best_model.pth'\n",
    "    checkpoint = torch.load(checkpoint_path, weights_only=False)\n",
    "    model.load_state_dict(checkpoint['model_state_dict'])\n",
    "    model = model.cuda() if torch.cuda.is_available() else model\n",
    "    model.eval()\n",
    "    \n",
    "    with torch.no_grad():\n",
    "        predictions, _ = model(tensor_data)\n",
    "        \n",
    "    predicted_classes = torch.argmax(predictions, dim=2).squeeze(0).cpu().numpy()\n",
    "    print(f\"Inference complete! Output shape: {predicted_classes.shape}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualization: Top-down view colored by S3DIS indoor predictions\n",
    "if 'predicted_classes' in locals():\n",
    "    s3dis_classes = [\"ceiling\", \"floor\", \"wall\", \"beam\", \"column\", \"window\", \"door\", \n",
    "                     \"table\", \"chair\", \"sofa\", \"bookcase\", \"board\", \"clutter\"]\n",
    "    \n",
    "    cmap = plt.get_cmap('tab20')\n",
    "    colors = cmap(predicted_classes / 13.0)[:, :3]\n",
    "    \n",
    "    fig = plt.figure(figsize=(10, 10))\n",
    "    ax = fig.add_subplot(111)\n",
    "    scatter = ax.scatter(xyz[:, 0], xyz[:, 1], c=colors, s=5, alpha=0.8)\n",
    "    plt.title(\"Top-Down LiDAR View (Colored by S3DIS Indoor Classes)\")\n",
    "    plt.axis('equal')\n",
    "    plt.show()\n",
    "    \n",
    "    print(\"Classes detected in this frame:\")\n",
    "    for c in np.unique(predicted_classes):\n",
    "        count = np.sum(predicted_classes == c)\n",
    "        print(f\"- {s3dis_classes[c]} (Class {c}): {count} points\")"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open("c:\\Users\\tina\\lidar-analysis\\Lidar-Analysis\\end_to_end_pipeline.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)
