import nbformat as nbf
nb = nbf.v4.new_notebook()

cell2 = nbf.v4.new_markdown_cell('# Process Video with YOLOv11 (First 30 Frames for Demo)')
cell2_code = nbf.v4.new_code_cell('''
import cv2
from ultralytics import YOLO
import os

model_path = '/Users/pavanreddy/Epilogue/YOLO_Inference/best_parinita.pt'
print(f'Loading model {model_path}...')
model = YOLO(model_path)

input_video = 'input_video.mp4'
output_video = 'annotated_youtube_video.mp4'

if not os.path.exists(input_video):
    print(f"Error: {input_video} not found!")
else:
    cap = cv2.VideoCapture(input_video)
    
    # Get video properties for writer
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Processing video: {w}x{h} @ {fps}fps | Limiting to 30 Frames to save time")
    
    # Use mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= 30:
            break
            
        # Run YOLO inference
        results = model.predict(frame, conf=0.25, verbose=False)
        
        # Plot bounding boxes on frame
        annotated_frame = results[0].plot()
        
        # Write to output video
        out.write(annotated_frame)
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count}/30 frames...")
            
    cap.release()
    out.release()
    print("✅ Video processing complete!")
''')

cell3 = nbf.v4.new_markdown_cell('# View Output Video\n*Note: HTML5 video embedding might not display in all notebook environments. The file is saved as `annotated_youtube_video.mp4`.*')
cell3_code = nbf.v4.new_code_cell('''
from IPython.display import Video
Video("annotated_youtube_video.mp4", embed=True, width=800)
''')

nb['cells'] = [cell2, cell2_code, cell3, cell3_code]
with open('/Users/pavanreddy/Epilogue/YOLO_Inference/youtube_yolo_inference.ipynb', 'w') as f:
    nbf.write(nb, f)
print('Notebook created!')
