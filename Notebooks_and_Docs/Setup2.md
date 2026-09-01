# Unreal Engine + Simulink Setup Guide

This guide covers exactly what you need to install and the exact steps to build your ego vehicle in Simulink and extract raw pictures using Unreal Engine.

## Phase 1: Installation (What you need to download)

Because you are using custom RoadRunner environments, you cannot use the default built-in viewer. You need the full Unreal Engine Editor.

1. **[Visual Studio 2022 (Free Community Edition)](https://aka.ms/vs/17/release/vs_community.exe)**
   * Why: Unreal Engine requires a C++ compiler to compile the MathWorks plugins.
   * During installation, you MUST check two workloads: **Desktop development with C++** and **Game development with C++**.
2. **Epic Games Launcher & Unreal Engine 5**
   * Download the Epic Games Launcher.
   * Go to the Unreal Engine tab -> Library -> Click the `+` to install an engine version.
   * **Crucial:** You must install the specific UE version supported by your MATLAB version. (e.g., MATLAB R2024a supports **UE 5.1 or 5.2**, *not* 5.4). Check the MathWorks documentation for your exact MATLAB release.
3. **MATLAB Add-Ons**
   * Open MATLAB. Go to `Home -> Add-Ons -> Get Add-Ons`.
   * Search for and install: **Automated Driving Toolbox Interface for Unreal Engine 5 Projects**.

---

## Phase 2: Connecting Unreal Engine and MATLAB

MathWorks provides a plugin that allows Simulink to talk to Unreal Engine. You have to copy this plugin into your Unreal project.

1. **Create an Unreal Project:** Open Unreal Engine, create a new blank "Games" project (C++ based, not Blueprint).
2. **Find the Plugin:** Open your Windows File Explorer and navigate to your MATLAB installation folder. It usually looks like this:
   `C:\Program Files\MATLAB\R2024a\toolbox\shared\sim3dprojects\epic\Plugins`
3. **Copy the Plugin:** Copy the folder named `MathWorksSimulation` (or similar).
4. **Paste the Plugin:** Go to your new Unreal Engine project folder. Create a folder named `Plugins` (if it doesn't exist) and paste the MathWorks folder inside.
5. **Open Unreal:** Double-click your `.uproject` file. It will ask to rebuild the plugins. Click Yes. Keep the Unreal Editor open.

---

## Phase 3: Building the Ego Vehicle in Simulink

Now we build the "brain" in Simulink that will control the car in the Unreal world.

1. **Create a Model:** Open MATLAB, type `simulink`, and create a **Blank Model**.
2. **Link to Unreal:**
   * Open the Library Browser. Search for **Simulation 3D Scene Configuration**. Drag it in.
   * Double-click it. Change *Scene source* to **Unreal Editor**.
   * Under *Project name*, browse and select your Unreal `.uproject` file.
3. **Add the Ego Vehicle:**
   * Search for **Simulation 3D Vehicle with Ground Following**. Drag it in.
   * Double-click it. Name it **EgoCar**.
4. **Make it Move (Testing):**
   * The vehicle block requires inputs (X, Y, Yaw). 
   * Add a **Constant** block (value: `0`) and connect it to Y and Yaw.
   * Add a **Ramp** block (slope: `5`) and connect it to X. When you run the simulation, the car will slowly drive forward.

---

## Phase 4: Getting the Raw Camera Data

1. **Add the Camera:**
   * Search for **Simulation 3D Camera** in the Library Browser. Drag it in.
   * Double-click it. Under *Parent Name*, type **EgoCar** (this glues the camera to the car).
   * Under the *Parameters* tab, set Image Size to `640 x 640`.
2. **View the Picture:**
   * To instantly see the picture in Simulink, search for a **Video Viewer** block.
   * Connect the output of the Camera to the Video Viewer.
3. **Run the Simulation:**
   * Click the **Run** button in Simulink. 
   * Simulink will talk to your open Unreal Engine editor. The car will spawn, drive forward, and your Video Viewer block will pop up showing the raw RGB picture of the Unreal world!
4. **Send to Python:**
   * Once you confirm the Video Viewer is showing the picture, you can delete the viewer and replace it with a **UDP Send** block to stream that image matrix directly to your Python YOLO team.