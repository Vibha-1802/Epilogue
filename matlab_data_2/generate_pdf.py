from fpdf import FPDF
import os

class PDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'Autonomous Driving Sensor Data Report', border=False, ln=True, align='C')
        self.set_font('helvetica', 'I', 10)
        self.cell(0, 10, 'Detailed Breakdown of matlab_data_2 Export', border=False, ln=True, align='C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('helvetica', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, border=False, ln=True, fill=True)
        self.ln(4)

    def chapter_body(self, text):
        self.set_font('helvetica', '', 11)
        self.multi_cell(0, 7, text)
        self.ln(5)

def create_report():
    print("Generating PDF Report...")
    pdf = PDF()
    pdf.add_page()
    
    # Overview
    pdf.chapter_title('1. Overview & Data Source')
    text = (
        "This report details the structural analysis of the 'matlab_data_2' directory. "
        "The data originates from a MATLAB Automated Driving Toolbox simulation, consisting of three main sensors: "
        "Radar, Camera, and LiDAR. Each sensor's output has been meticulously analyzed and reformatted into "
        "Python-accessible structures."
    )
    pdf.chapter_body(text)
    
    # Radar
    pdf.chapter_title('2. Radar Data (r_data.mat)')
    text = (
        "Format: MATLAB Struct (.mat)\n"
        "Contents: 50 independent radar detections.\n\n"
        "Fields Extracted:\n"
        "- Time: Simulation timestamp of detection.\n"
        "- SensorIndex: Unique ID of the radar sensor.\n"
        "- ObjectClassID: Numerical classification of the detected object.\n"
        "- Measurement: Numerical array typically containing [Azimuth, Range, Range_Rate].\n\n"
        "Interpretation: The 'Measurement' field provides crucial spatial and velocity metrics (Range and Range_Rate) "
        "which are used to calculate the Time-to-Collision (TTC) for risk scoring algorithms."
    )
    pdf.chapter_body(text)
    
    # Camera
    pdf.chapter_title('3. Camera Data (c_data.mat)')
    text = (
        "Format: MATLAB Struct (.mat)\n"
        "Contents: 50 independent camera object detections.\n\n"
        "Crucial Note: Unlike traditional camera exports, this file does NOT contain raw RGB pixel arrays or "
        "video frames. The simulation exported processed 'Camera Detections' (e.g., bounding boxes and class IDs). "
        "Because it lacks raw pixels, image frames cannot be reconstructed.\n\n"
        "Fields Extracted:\n"
        "- Identical structure to radar data, containing Time, SensorIndex, ObjectClassID, and Measurement.\n"
        "- Measurement typically contains 2D Bounding Box coordinates [x, y, width, height]."
    )
    pdf.chapter_body(text)
    
    # Lidar
    pdf.chapter_title('4. LiDAR Data (Lidar_PCD_Data/)')
    text = (
        "Format: Standard Point Cloud Data (.pcd) Files\n"
        "Contents: 800 individual frames of 3D spatial points.\n\n"
        "Data Dictionary (Per Point):\n"
        "- X, Y, Z: Euclidean spatial coordinates in meters relative to the ego-vehicle.\n\n"
        "Interpretation: Because these are standard .pcd files, they can be natively ingested by Python using "
        "the 'open3d' library. To utilize this data for trajectory prediction, the points must first undergo "
        "RANSAC Ground Plane Removal, followed by DBSCAN clustering to group points into distinct 3D objects."
    )
    pdf.chapter_body(text)
    
    # Conclusion
    pdf.chapter_title('5. Conclusion & Actionable Steps')
    text = (
        "The proprietary MATLAB files have been successfully unpacked. The radar and camera detections have been "
        "extracted into clean CSV spreadsheets located in the 'Results/' directory. The 800 LiDAR .pcd files are "
        "ready for immediate Python processing via Open3D."
    )
    pdf.chapter_body(text)
    
    os.makedirs("Results", exist_ok=True)
    output_path = 'Results/Sensor_Data_Report.pdf'
    pdf.output(output_path)
    print(f"✅ Report generated successfully: {output_path}")

if __name__ == '__main__':
    create_report()
