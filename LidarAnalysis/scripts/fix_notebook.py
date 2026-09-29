import nbformat

file_path = '/Users/pavanreddy/Documents/MATLAB/LidarAnalysis/scripts/04_Cluster_Video_Visualization.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code':
        if 'data_dir =' in cell.source and 'PLAYBACK_SPEED' in cell.source:
            # Replace the path setup block
            new_source = []
            for line in cell.source.splitlines(True):
                if line.startswith('data_dir = '):
                    new_source.append('data_dir = "../data/raw/lidar_with_intensity_and_clusters"\n')
                elif 'if not os.path.exists(data_dir)' in line:
                    continue
                elif '    data_dir = os.path.join' in line:
                    continue
                else:
                    # fix the weird syntax error from previous run
                    if line.strip() == r'data_dir = "../data/raw/lidar_with_intensity_and_clusters"\n':
                        continue
                    new_source.append(line)
            cell.source = "".join(new_source)

with open(file_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Notebook updated successfully.")
