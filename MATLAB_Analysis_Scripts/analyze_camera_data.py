import scipy.io
import numpy as np
import cv2
import os

def main():
    print("📸 Analyzing camera_data.mat...")
    try:
        data = scipy.io.loadmat("camera_data.mat", squeeze_me=True, struct_as_record=False)
    except Exception as e:
        print(f"❌ Failed to load camera_data.mat: {e}")
        return

    cam = data["camera_matrix"]
    
    # Shape is typically (Height, Width, Channels, Frames) from MATLAB
    print(f"Original MATLAB shape: {cam.shape}")
    
    h, w, c, num_frames = cam.shape
    
    output_dir = "camera_frames"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # 5.0 FPS is a good default for low-frame-count simulations
    out = cv2.VideoWriter('camera_video.mp4', fourcc, 5.0, (w, h))
    
    for i in range(num_frames):
        # Extract the i-th frame
        frame = cam[:, :, :, i]
        
        # Oops, fixing the cv2 call:
        if frame.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
            
        # Save as individual image
        img_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(img_path, frame_bgr)
        
        # Write to MP4 video
        out.write(frame_bgr)
        
    out.release()
    print(f"\n✅ Extracted {num_frames} frames into '{output_dir}/' folder.")
    print("✅ Compiled all frames into 'camera_video.mp4' video file.")

if __name__ == "__main__":
    main()
