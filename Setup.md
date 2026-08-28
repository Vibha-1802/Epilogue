# Ultimate Guide: Connecting MATLAB/Simulink R2026a to Unreal Engine 5.4

This guide covers the entire, step-by-step process of integrating Simulink with a custom Unreal Engine 5.4 project for Autonomous Vehicle simulation. It specifically addresses all the undocumented bugs, hidden folders, and compilation errors you will encounter along the way.

---

## Phase 1: Required Software Installations
Before doing anything, ensure your PC has the following installed:

1. **MATLAB R2026a** 
   - Ensure you have the **Automated Driving Toolbox** installed.
2. **Visual Studio 2022 Community**
   - When installing, you MUST select the **"Game development with C++"** workload. Without this, Unreal Engine cannot compile the MathWorks plugin.
3. **Unreal Engine 5.4**
   - Installed via the Epic Games Launcher.
4. **The MathWorks Unreal Engine Add-On**
   - Open MATLAB.
   - Go to the **Home** tab -> **Add-Ons** -> **Get Add-Ons**.
   - Search for: `Automated Driving Toolbox Interface for Unreal Engine 5 Projects`.
   - Click **Install**. *(If you don't do this, the plugin files will literally not exist on your hard drive).*

---

## Phase 2: Locating the Hidden Plugin
MathWorks buries the Unreal Engine plugin deep inside a hidden system folder.

1. Open Windows File Explorer.
2. Click the address bar at the very top, paste this exact path, and press Enter:
   `C:\ProgramData\MATLAB\SupportPackages\R2026a\toolbox\shared\sim3dprojects\spkg\plugins\mw_simulation`
   *(Note: `ProgramData` is a hidden folder, so you must paste the path directly).*
3. You will see a folder named **`MathWorksSimulation`**. **Copy** this folder (do not cut/move it).

---

## Phase 3: Setting Up Your Unreal Project
1. Open Unreal Engine 5.4 and create your custom project (e.g., `MyProject`).
2. Navigate to your new project folder in Windows File Explorer (where your `.uproject` file is).
3. Create a new folder named exactly **`Plugins`** (with an "s").
4. **Paste** the `MathWorksSimulation` folder inside the `Plugins` folder.

---

## Phase 4: Fixing the MathWorks Bugs (CRITICAL)
If you try to compile the project now, it will fail. MathWorks R2026a has a few bugs when porting to UE 5.4: it forgets where MATLAB is installed, and the Drone (UAV) physics modules fail to compile. Since we are building cars, we must disable the drones.

### Bug 1: Fix the Environment Variable
The compiler needs to know where MATLAB is installed.
1. Press the Windows Key, search for **PowerShell**, right-click it, and select **Run as Administrator**.
2. Paste this exact command and hit Enter:
   `[Environment]::SetEnvironmentVariable("MATLABROOT", "C:\Program Files\MATLAB\R2026a", "User")`
3. This permanently tells the compiler where MATLAB is.

### Bug 2: Disable Broken Drone/Raytracing Modules
1. Go to your Unreal project: `Plugins\MathWorksSimulation\`.
2. Right-click the **`MathWorksSimulation.uplugin`** file and select **Open with -> Notepad**.
3. Scroll down to the `"Modules": [` section.
4. Carefully **DELETE** the entire blocks (from `{` to `}`) for the following three modules:
   - `"Name": "MathWorksRaytracing"`
   - `"Name": "MathWorksAerospace"`
   - `"Name": "MathWorksUAV"`
5. Save the file and close Notepad.
*(This prevents the compiler from crashing on drone physics you don't need).*

---

## Phase 5: Compiling the Plugin
1. **CRITICAL:** Ensure Unreal Engine and Visual Studio are COMPLETELY CLOSED. If Unreal is open, "Live Coding" will lock the files and the build will fail with Exit Code 1.
2. In File Explorer, go to your Unreal project folder.
3. **Right-click** your `.uproject` file.
4. Click **Generate Visual Studio project files**. (A quick black loading bar will appear).
5. Now, **Double-click** your `.uproject` file to open it.
6. A pop-up will warn you that the MathWorks modules are missing/out-of-date and ask if you want to rebuild them. Click **Yes**.
7. Wait 30-60 seconds. The project will open successfully!

---

## Phase 6: Wiring Up Simulink
Now that Unreal Engine is open and ready, let's build the brain.

1. Open MATLAB and type `simulink` to create a Blank Model.
2. Open the **Library Browser** and find the **Automated Driving Toolbox -> Simulation 3D** section.
3. Drag the following 5 blocks onto your canvas:
   - `Simulation 3D Scene Configuration`
   - `Simulation 3D Vehicle with Ground Following`
   - `Simulation 3D Camera`
   - `Simulation 3D Lidar`
   - `Simulation 3D Probabilistic Radar`

**Configuration:**
1. **Link to Unreal:** Double-click the `Scene Configuration` block. Set *Scene source* to **Unreal Editor** and select your `.uproject` file.
2. **Name the Car:** Double-click the `Vehicle` block. Change the *Name* field to **EgoCar**.
3. **Mount Sensors:** Double-click the Camera, Lidar, and Radar blocks. Set their *Parent Name* to exactly **EgoCar**. (For the camera, set the resolution to 640x640 for YOLO).
4. **Make it Move:** Add a `Ramp` block and connect it to the Vehicle's `X` input. Add a `Constant` block (set to `0`) and connect it to `Y` and `Yaw`.
5. **View Output:** Add a `Video Viewer` block and connect it to the Camera output.

**Final Step:** Hit the green **RUN** button in Simulink. It will connect to Unreal Engine, the car will drive, and your camera feed will appear!

