from fpdf import FPDF
import os

class PDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'Autonomous Driving Sensor Data Report (Dataset 3)', border=False, ln=True, align='C')
        self.set_font('helvetica', 'I', 10)
        self.cell(0, 10, 'Detailed Breakdown of matlab_data_3 Export', border=False, ln=True, align='C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('helvetica', 'B', 12)
        self.set_fill_color(255, 220, 200) # Slightly different color to distinguish
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
    pdf.chapter_title('1. Overview & Data Source Variations')
    text = (
        "This report details the structural analysis of the 'matlab_data_3' directory. "
        "Unlike previous datasets, this simulation run was exported with a distinct variation: "
        "Camera data was completely omitted. This forces the downstream algorithms to rely purely on "
        "Radar and LiDAR for object detection and trajectory prediction."
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
        "Interpretation: Since camera data is missing, the Radar acts as the primary source for "
        "calculating relative velocity (Range_Rate) to compute Time-to-Collision (TTC)."
    )
    pdf.chapter_body(text)
    
    # Camera
    pdf.chapter_title('3. Camera Data (MISSING)')
    text = (
        "Status: OMitted / Not Exported.\n"
        "The c_data.mat file is completely absent from this dataset. Autonomous Emergency Braking (AEB) "
        "systems handling this dataset must implement Sensor Fusion algorithms that combine Radar (for speed) "
        "and LiDAR (for volumetric size/bounding boxes) without optical classification confirmation."
    )
    pdf.chapter_body(text)
    
    # Lidar
    pdf.chapter_title('4. LiDAR Data (Lidar_PCD_Data/)')
    text = (
        "Format: Standard Point Cloud Data (.pcd) Files\n"
        "Contents: 1001 individual frames of 3D spatial points.\n\n"
        "Data Volume: The presence of 1001 frames (compared to 800 in previous sets) indicates either a "
        "longer simulation scenario or a higher frequency capture rate.\n\n"
        "Processing Pipeline: Python's 'open3d' library should be used to load the point clouds, apply "
        "RANSAC to remove the ground plane, and DBSCAN to cluster points into 3D Bounding Boxes to compensate "
        "for the missing optical camera detections."
    )
    pdf.chapter_body(text)
    
    # Conclusion
    pdf.chapter_title('5. Conclusion & Actionable Steps')
    text = (
        "The proprietary MATLAB files have been unpacked. The radar detections have been extracted into "
        "Results/radar_detections.csv. The 1001 LiDAR .pcd files are ready for immediate Python processing. "
        "Algorithms built for this dataset must account for the lack of camera telemetry."
    )
    pdf.chapter_body(text)
    
    os.makedirs("Results", exist_ok=True)
    output_path = 'Results/Sensor_Data_Report_3.pdf'
    pdf.output(output_path)
    print(f"✅ Report generated successfully: {output_path}")

if __name__ == '__main__':
    create_report()
