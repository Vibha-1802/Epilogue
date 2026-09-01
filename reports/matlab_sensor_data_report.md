# MATLAB Sensor Data Analysis Report

This report provides a comprehensive breakdown of the output files located in your `matlab_data_2` folder: `c_data.mat` and `r_data.mat`. Both files export data using MATLAB's `objectDetection` struct array format, but they capture entirely different physical phenomena based on their respective sensors.

---

## 1. Vision Detector Data (`c_data.mat`)

The `c_data.mat` file contains the output from the **Vision Detector Generator**. This simulates a forward-facing optical camera combined with a computer vision algorithm (like a YOLO or SSD model) that analyzes pixels to identify objects.

### Extracted Fields & Definitions:

* **`Time`**
  * **What it is:** The exact timestamp (in seconds) within the simulation when the camera captured the frame and made the detection.
  * **Why it matters:** Essential for synchronizing with other sensors (like Radar and LiDAR) and for calculating object trajectories over time.

* **`SensorIndex`**
  * **What it is:** An integer ID assigned to this specific camera.
  * **Why it matters:** If your vehicle has multiple cameras (e.g., front, rear, left, right), this index tells you which camera saw the object.

* **`ObjectClassID`**
  * **What it is:** A numerical label assigned to the object based on visual classification (e.g., `1` = Car, `2` = Pedestrian, `3` = Bicycle).
  * **Why it matters:** This tells your system *what* it is looking at. Optical cameras are excellent at classification but poor at judging exact distance.

* **`Measurement` (Bounding Box)**
  * **What it is:** For a vision sensor, the measurement is a 1D array of 4 numerical values representing a 2D Bounding Box: `[x_top_left, y_top_left, width, height]`.
  * **Why it matters:** This defines the rectangular boundary of the object within the 2D image plane (measured in pixels). It tells your system where the object is located on the screen, but it does *not* provide real-world 3D depth or distance.

* **`MeasurementNoise`**
  * **What it is:** A 4x4 Covariance Matrix representing the statistical uncertainty of the bounding box coordinates.
  * **Why it matters:** Computer vision isn't perfect. This matrix is fed into tracking filters (like a Kalman Filter) to smooth out jittery bounding boxes over multiple frames.

---

## 2. Radar Sensor Data (`r_data.mat`)

The `r_data.mat` file contains the output from the **Radar Sensor**. Unlike a camera, Radar does not "see" pixels or colors; it emits radio waves and listens for their reflection. It is terrible at classifying objects, but excellent at measuring kinematics (physics).

### Extracted Fields & Definitions:

* **`Time`**
  * **What it is:** The simulation timestamp (in seconds) of the radar ping.

* **`SensorIndex`**
  * **What it is:** The integer ID of the Radar sensor emitting the waves.

* **`ObjectClassID`**
  * **What it is:** A numerical label for the object. 
  * **Why it matters:** In Radar, this is often a generic identifier (e.g., "Unknown Object" or a general cluster ID) because Radar cannot distinguish between a car and a large metal dumpster based purely on radio reflections.

* **`Measurement` (Kinematics)**
  * **What it is:** For a Radar sensor, the measurement is a 1D array of 3 numerical values representing spherical physics data: `[Azimuth, Range, RangeRate]`.
  
  * **Field Breakdown:**
    1. **`Azimuth` (Angle):** The horizontal angle from the center of the radar (bore-sight) to the detected object. It tells your system if the object is directly in front (0 degrees), to the left (negative degrees), or to the right (positive degrees).
    2. **`Range` (Distance):** The direct, straight-line distance from your vehicle to the object, usually measured in meters.
    3. **`RangeRate` (Relative Velocity):** The speed at which the object is moving *relative to your vehicle*. 
       * A **negative** RangeRate means the object is getting closer (approaching). 
       * A **positive** RangeRate means the object is moving away (receding).
       * *Crucial Use:* This value, combined with Range, is used to calculate Time-To-Collision (TTC) for Automatic Emergency Braking (AEB).

* **`MeasurementNoise`**
  * **What it is:** A 3x3 Covariance Matrix representing the statistical uncertainty of the Azimuth, Range, and RangeRate.
  * **Why it matters:** Radar signals can bounce off multiple surfaces (multipath interference). This matrix helps your Kalman Filter trust or distrust certain radar pings based on their noise levels.

---

## Summary of Sensor Fusion
Because `c_data` gives you excellent **Classification** (What is it?) and `r_data` gives you excellent **Physics** (How far is it, and how fast is it moving?), your downstream algorithms must synchronize these files by `Time` and merge the measurements. This allows your vehicle to know: *"That object is a Car (from Vision), it is 45 meters away (from Radar), and it is approaching us at 15 m/s (from Radar)."*
