import cv2

video1_path = '/Users/pavanreddy/Epilogue/YOLO_Inference/annotated_youtube_video.mp4'
video2_path = '/Users/pavanreddy/Epilogue/YOLO_Inference/annotated_potholes_3.mp4'
output_path = '/Users/pavanreddy/Epilogue/YOLO_Inference/demo_real_video.mp4'

cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

if not cap1.isOpened() or not cap2.isOpened():
    print("Error opening video files!")
    exit(1)

fps = cap1.get(cv2.CAP_PROP_FPS)
if fps == 0: fps = 30.0
width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frames_per_video = int(5 * fps)

print(f"Reading {frames_per_video} frames from video 1...")
count1 = 0
while count1 < frames_per_video:
    ret, frame = cap1.read()
    if not ret: break
    out.write(frame)
    count1 += 1

print(f"Reading {frames_per_video} frames from video 2...")
count2 = 0
while count2 < frames_per_video:
    ret, frame = cap2.read()
    if not ret: break
    # Resize to match first video's dimensions
    resized_frame = cv2.resize(frame, (width, height))
    out.write(resized_frame)
    count2 += 1

cap1.release()
cap2.release()
out.release()
print(f"✅ Successfully combined videos into {output_path}")
