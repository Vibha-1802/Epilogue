from fpdf import FPDF
import os

class PDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 16)
        self.cell(0, 10, 'Master Autonomous Driving Data Specification (For LLM Prompting)', border=False, ln=True, align='C')
        self.set_font('helvetica', 'I', 11)
        self.cell(0, 10, 'Comprehensive format guide for generating Risk Scoring & Trajectory Algorithms', border=False, ln=True, align='C')
        self.ln(5)

    def chapter_title(self, title):
        self.set_font('helvetica', 'B', 13)
        self.set_fill_color(230, 240, 255)
        self.cell(0, 10, title, border=False, ln=True, fill=True)
        self.ln(4)

    def chapter_body(self, text):
        self.set_font('helvetica', '', 11)
        self.multi_cell(0, 6, text)
        self.ln(4)

def create_master_report():
    print("Generating Master PDF Report...")
    pdf = PDF()
    pdf.add_page()
    
    # 1. System Overview
    pdf.chapter_title('1. System Architecture & Objective')
    text = (
        "Objective: Generate Python algorithms for Object Detection, Trajectory Prediction, and "
        "Collision Avoidance (AEB) using a multi-sensor autonomous driving setup.\n\n"
        "Sensors Provided from Simulation:\n"
        "1. Camera Raw Frames (.jpg images)\n"
        "2. Camera Object Detections (Classifications & Bounding Boxes)\n"
        "3. Radar Detections (Kinematic Physics Data)\n"
        "4. LiDAR Point Clouds (.pcd files containing 3D spatial points)\n\n"
        "The algorithms generated must accurately ingest these specific data formats to perform Sensor Fusion."
    )
    pdf.chapter_body(text)
    
    # 2. Camera Raw Frames
    pdf.chapter_title('2. Data Format: Camera Raw Frames')
    text = (
        "Source Type: Standard JPEG image sequence (.jpg).\n"
        "Resolution: Typically 1280x720 RGB.\n\n"
        "Algorithm Requirements (Object Detection):\n"
        "- The raw frames must be ingested by an optical inference model (e.g., YOLO11) to detect "
        "bounding boxes [x, y, w, h] and class confidences for autonomous driving classes (Car, Pedestrian, etc.).\n"
        "- OpenCV (cv2) should be used for frame ingestion and preprocessing."
    )
    pdf.chapter_body(text)

    # 3. Camera Detections
    pdf.chapter_title('3. Data Format: Camera Object Detections')
    text = (
        "Source Type: Extracted CSV from MATLAB 'c_data.mat'.\n"
        "Fields Available:\n"
        "- Time: Simulation timestamp.\n"
        "- SensorIndex: Camera ID.\n"
        "- ObjectClassID: Integer representing the class (e.g., 1=Car).\n"
        "- Measurement: 2D Bounding Box array [X_top_left, Y_top_left, Width, Height].\n\n"
        "Algorithm Requirements (Ground Truth / Fusion):\n"
        "- This data represents the simulation's built-in perception output. It should be used to validate "
        "the custom Object Detection model or fused with Radar to assign class identities to radar tracks."
    )
    pdf.chapter_body(text)
    
    # 4. Radar Detections
    pdf.add_page()
    pdf.chapter_title('4. Data Format: Radar Kinematics')
    text = (
        "Source Type: Extracted CSV from MATLAB 'r_data.mat'.\n"
        "Fields Available:\n"
        "- Time: Simulation timestamp.\n"
        "- SensorIndex: Radar ID.\n"
        "- ObjectClassID: Integer representing the class (often generic for radar).\n"
        "- Measurement: Numerical array containing physics metrics [Azimuth, Range, Range_Rate].\n\n"
        "Definitions:\n"
        "- Range (meters): Distance to the object.\n"
        "- Azimuth (degrees/radians): Angle to the object relative to ego-vehicle.\n"
        "- Range_Rate (m/s): Relative velocity of the object (negative = approaching, positive = receding).\n\n"
        "Algorithm Requirements (Trajectory & Collision):\n"
        "- Time-To-Collision (TTC) must be calculated dynamically: TTC = Range / ABS(Range_Rate) (only if approaching).\n"
        "- Constant Velocity (CV) Trajectory Prediction should use the Azimuth and Range_Rate to predict "
        "future X/Y coordinates of the object over a 1 to 3-second horizon."
    )
    pdf.chapter_body(text)
    
    # 5. LiDAR Point Clouds
    pdf.chapter_title('5. Data Format: LiDAR Point Clouds')
    text = (
        "Source Type: Sequence of standard .pcd (Point Cloud Data) files.\n"
        "Fields Available (Per Point):\n"
        "- X, Y, Z: 3D spatial coordinates in meters relative to the ego-vehicle.\n\n"
        "Algorithm Requirements (Clustering & Validation):\n"
        "- Must use Python 'open3d' library to ingest .pcd files.\n"
        "- Step 1: Apply RANSAC (Random Sample Consensus) to segment and remove the ground plane (the road).\n"
        "- Step 2: Apply DBSCAN (Density-Based Spatial Clustering) on remaining points to group them into solid objects.\n"
        "- Step 3: Compute Axis-Aligned Bounding Boxes (AABB) for each cluster to determine volumetric size and center (X_c, Y_c, Z_c).\n"
        "- Step 4: Fuse this 3D centroid data with the Radar data to confirm physical presence and distance."
    )
    pdf.chapter_body(text)
    
    # 6. Final LLM Prompting Instructions
    pdf.chapter_title('6. Instructions for Algorithm Formulation')
    text = (
        "When generating the Python scripts, the LLM MUST adhere to the following logic pipeline:\n\n"
        "1. Perception Phase: Run YOLO on Camera JPEG frames to identify visual bounding boxes and classes.\n"
        "2. Clustering Phase: Process simultaneous LiDAR .pcd frames with DBSCAN to find 3D geometric clusters.\n"
        "3. Sensor Fusion Phase: Match YOLO 2D bounding boxes and LiDAR 3D clusters with the incoming Radar "
        "measurements (using spatial heuristics) to assign a Range and Range_Rate to the detected objects.\n"
        "4. Risk Scoring Phase: For each fused object, calculate TTC. \n"
        "   - If TTC < 1.5s: Trigger Autonomous Emergency Braking (CRITICAL RISK).\n"
        "   - If TTC < 3.0s: Trigger Driver Warning (HIGH RISK).\n"
        "   - Else: Monitor (SAFE).\n"
        "5. Trajectory Prediction: Implement a Kalman Filter or Constant Velocity model updating the predicted "
        "path of the object across sequential timestamps."
    )
    pdf.chapter_body(text)

    output_path = '/Users/pavanreddy/Epilogue/Master_Autonomous_Driving_Spec.pdf'
    pdf.output(output_path)
    print(f"✅ Master Specification PDF generated successfully: {output_path}")

if __name__ == '__main__':
    create_master_report()
