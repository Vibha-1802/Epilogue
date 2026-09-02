import os
import sys
import cv2
import math
import numpy as np
from ultralytics import YOLO
import logging

logging.getLogger('ultralytics').setLevel(logging.ERROR)

sys.path.append('/Users/pavanreddy/Epilogue/object_risk_score')
import risk_score_calculator

model_path = '/Users/pavanreddy/Epilogue/YOLO_Inference/best_parinita.pt'
model = YOLO(model_path)

input_video = '/Users/pavanreddy/Epilogue/YOLO_Inference/matlab_data_4/cam_video.mp4'
output_video = '/Users/pavanreddy/Epilogue/YOLO_Inference/matlab_data_4/annotated_cam_video.mp4'

cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0: fps = 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

print('Processing video and rendering annotations...')
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    
    results = model.predict(frame, conf=0.25, verbose=False)
    boxes = results[0].boxes
    
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cls_id = int(box.cls[0].cpu().numpy())
        cls_name = model.names[cls_id]
        
        w = x2 - x1
        mock_dist = max(3.0, 1500.0 / (w + 1.0)) 
        img_center_x = width / 2
        obj_center_x = (x1 + x2) / 2
        mock_azimuth = (obj_center_x - img_center_x) / img_center_x * 30.0 
        mock_rate = -5.0 
        
        azimuth_rad = math.radians(mock_azimuth)
        x_current = mock_dist * math.cos(azimuth_rad)
        y_current = mock_dist * math.sin(azimuth_rad)
        vx = mock_rate * math.cos(azimuth_rad)
        vy = mock_rate * math.sin(azimuth_rad)
        
        risk_score = risk_score_calculator.calculate_risk(x_current, y_current, vx, vy)
        
        color = (0, 0, 255) if risk_score > 10 else (0, 255, 0)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f'{cls_name} | Risk: {risk_score:.1f}'
        cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
    out.write(frame)
    frame_idx += 1
    if frame_idx % 30 == 0:
        print(f'Processed {frame_idx} frames...')

cap.release()
out.release()
print(f'✅ Video saved to {output_video}')
