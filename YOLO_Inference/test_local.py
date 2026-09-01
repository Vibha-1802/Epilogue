import os
import sys
import time
from ultralytics import YOLO

def main():
    print("🚀 YOLO Local Inference Script")
    print("---------------------------------")
    
    # 1. Ask the user for the paths
    model_path = input("Enter the path to your downloaded best.pt file: ").strip().strip("'\"")
    image_path = input("Enter the path to the image you want to test: ").strip().strip("'\"")
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at '{model_path}'")
        sys.exit(1)
        
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at '{image_path}'")
        sys.exit(1)
        
    # 2. Load the model
    print(f"\nLoading model from {model_path}...")
    model = YOLO(model_path)
    
    # GPU Warmup (The first prediction is always artificially slow while memory allocates)
    print("Warming up Apple GPU (MPS)...")
    model.predict(source=image_path, imgsz=640, device='mps', verbose=False)
    
    # 3. Run Inference & Measure Time
    print(f"\nRunning fast inference on {image_path}...")
    
    # I increased conf to 0.45 to prevent wild guessing on messy images!
    results = model.predict(source=image_path, conf=0.45, imgsz=640, device='mps', verbose=False)
    
    # Ultralytics natively tracks the exact hardware time spent purely on inference!
    inference_time = results[0].speed['inference']
    print(f"\n⚡ Pure GPU Inference Time: {inference_time:.2f} milliseconds!")
    
    # 4. Display the results in a native popup window
    print("✅ Inference complete! Opening result window...")
    results[0].show()
    
    # 5. Print a quick text summary of what it found
    print("\n--- Objects Detected ---")
    names = results[0].names
    if len(results[0].boxes) == 0:
        print("No objects detected above confidence threshold.")
    else:
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"- {names[class_id]} ({conf:.2f})")

if __name__ == "__main__":
    main()
