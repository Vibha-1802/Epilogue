from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import os

app = FastAPI()

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_parinita.pt")
print(f"Loading YOLO model from {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("Model loaded successfully!")

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # Read image from request
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Run YOLO inference
    results = model.predict(img, conf=0.25, verbose=False)
    
    # Plot annotations (boxes, masks, labels) on the image
    # YOLO's .plot() method automatically handles masks if it's a segmentation model
    annotated_img = results[0].plot()
    
    # Encode back to JPEG
    _, buffer = cv2.imencode('.jpg', annotated_img)
    base64_str = base64.b64encode(buffer).decode('utf-8')
    
    return JSONResponse(content={
        "annotated_image": base64_str,
        "message": "Image analyzed successfully!"
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
