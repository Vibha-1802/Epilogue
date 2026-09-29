import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Styling for PPT
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['figure.facecolor'] = 'white'

def load_manifest():
    path = "../outputs/road_label_test/manifest.json"
    with open(path, 'r') as f:
        return json.load(f)

def generate_memory_chart(manifest):
    # Data from manifest
    mem_data = manifest['aggregate']['memory_occupied_only']
    
    methods = ['Raw Points\n(Baseline)', 'Uniform 3D Voxel\n(Sparse CNN)', 'Foveated 2.5D\n(Our Model)']
    
    # Get bytes used (convert to MB)
    raw_mb = mem_data['raw_points']['bytes'] / (1024**2)
    voxel_mb = mem_data['voxel_sparse']['bytes'] / (1024**2)
    fov_mb = mem_data['foveated_2_5d']['bytes'] / (1024**2)
    
    values = [raw_mb, voxel_mb, fov_mb]
    colors = ['#ff4d4d', '#ffaa00', '#4CAF50']
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    bars = ax.bar(methods, values, color=colors, width=0.6, edgecolor='black', linewidth=1.2)
    
    ax.set_title("RAM Usage: Foveated 2.5D vs Baselines (Per Frame)", fontsize=22, fontweight='bold', pad=20)
    ax.set_ylabel("Memory Allocation (MB)", fontsize=16, fontweight='bold')
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"{yval:.2f} MB", 
                ha='center', va='bottom', fontsize=16, fontweight='bold')
                
    # Add percentage reduction annotation
    reduction = (1 - fov_mb / voxel_mb) * 100
    ax.text(1.5, voxel_mb * 0.8, f"{reduction:.1f}% Reduction\nvs Sparse Voxel", 
            fontsize=16, fontweight='bold', color='darkgreen', 
            bbox=dict(facecolor='#e6ffe6', edgecolor='green', boxstyle='round,pad=0.5'))

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('ppt_ram_usage_results.png')
    plt.close()

def generate_latency_chart(manifest):
    lat = manifest['aggregate']['latency_ms']
    
    # Extract stages
    stages = ['preprocess', 'index', 'semantics', 'freespace', 'terrain', 'instances', 'plan']
    stage_names = ['Preprocess', 'Grid Indexing', 'Semantics', 'Free-Space', 'Terrain', 'Obj Cluster', 'A* Planning']
    
    values = [lat[s]['mean'] for s in stages]
    
    # Filter out 0 values to keep chart clean
    filtered = [(n, v) for n, v in zip(stage_names, values) if v > 0]
    names, vals = zip(*filtered)
    
    # Pastel colors
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99', '#c2c2f0','#ffb3e6', '#c4e17f']
    
    fig, ax = plt.subplots(figsize=(9, 7), dpi=200)
    
    def func(pct, allvals):
        absolute = int(np.round(pct/100.*np.sum(allvals)))
        return f"{pct:.1f}%\n({absolute}ms)"
        
    wedges, texts, autotexts = ax.pie(vals, labels=names, autopct=lambda pct: func(pct, vals),
                                      startangle=140, colors=colors[:len(vals)],
                                      textprops=dict(color="black", fontweight='bold', fontsize=12),
                                      wedgeprops=dict(edgecolor='w', linewidth=1.5))
                                      
    ax.set_title("Pipeline CPU Compute Breakdown", fontsize=22, fontweight='bold', pad=20)
    
    total_ms = sum(vals)
    fps = 1000.0 / total_ms
    
    # Annotate total FPS
    ax.text(0, -1.3, f"Total Latency: {total_ms:.1f} ms\nThroughput: {fps:.1f} FPS", 
            ha='center', fontsize=16, fontweight='bold', 
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
            
    plt.tight_layout()
    plt.savefig('ppt_compute_latency.png')
    plt.close()

def generate_disk_usage_chart():
    # We will manually calculate the total size of the PCDs vs the serialized BINs
    data_dir = "../data/labeled/road_label_test"
    bin_dir = "../outputs/road_label_test/cells"
    
    pcd_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pcd')]
    bin_files = [os.path.join(bin_dir, f) for f in os.listdir(bin_dir) if f.endswith('.bin')]
    
    pcd_size_mb = sum(os.path.getsize(f) for f in pcd_files) / (1024**2)
    bin_size_mb = sum(os.path.getsize(f) for f in bin_files) / (1024**2)
    
    methods = ['Raw LiDAR Dataset\n(.PCD Files)', 'Foveated Serialized\n(.BIN Format)']
    values = [pcd_size_mb, bin_size_mb]
    colors = ['#e74c3c', '#2ecc71']
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    bars = ax.barh(methods, values, color=colors, height=0.5, edgecolor='black', linewidth=1.2)
    
    ax.set_title("Disk Storage Footprint (Entire Sequence)", fontsize=22, fontweight='bold', pad=20)
    ax.set_xlabel("Total File Size (MB)", fontsize=16, fontweight='bold')
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 2, bar.get_y() + bar.get_height()/2, f"{width:.1f} MB", 
                ha='left', va='center', fontsize=16, fontweight='bold')
                
    compression = (pcd_size_mb / bin_size_mb)
    
    ax.text(pcd_size_mb * 0.5, 0.5, f"{compression:.1f}x Compression Ratio!", 
            fontsize=18, fontweight='bold', color='white', 
            bbox=dict(facecolor='darkblue', edgecolor='blue', boxstyle='round,pad=0.5'))
            
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('ppt_disk_storage.png')
    plt.close()

if __name__ == "__main__":
    m = load_manifest()
    generate_memory_chart(m)
    generate_latency_chart(m)
    generate_disk_usage_chart()
    print("Generated all 3 PPT slides!")
