% process_sensor_data.m
% This script loads Radar, Camera, and LiDAR data, processes it (closed-loop),
% and outputs a .mat file containing timeseries objects for Simulink.

clc; clear; close all;

%% 1. Load Sensor Data
disp('Loading Radar Data...');
% Make sure this script is in the same folder as the .mat files
if isfile('r_data.mat')
    radar_data = load('r_data.mat'); 
else
    warning('r_data.mat not found.');
end

disp('Loading Camera Data...');
if isfile('ca_data.mat')
    camera_data = load('ca_data.mat'); 
else
    warning('ca_data.mat not found.');
end

disp('Loading LiDAR PCD Data...');
lidarFolder = 'Lidar_PCD_Data';
if isfolder(lidarFolder)
    ptCloudFiles = dir(fullfile(lidarFolder, '*.pcd'));
    disp(['Found ', num2str(length(ptCloudFiles)), ' LiDAR PCD files.']);
    
    % Example of how to read the first PCD file in a loop:
    % if ~isempty(ptCloudFiles)
    %     ptCloud = pcread(fullfile(lidarFolder, ptCloudFiles(1).name));
    % end
else
    warning('Lidar_PCD_Data folder not found.');
end

%% 2. Process Data & Closed-Loop Control Logic
disp('Processing sensor data to calculate vehicle controls...');

% Define the simulation timeline (e.g., 0 to 10 seconds with 0.1s step)
dt = 0.1;
t = (0:dt:10)'; 
num_steps = length(t);

% Initialize control arrays
delta = zeros(num_steps, 1); % Steering Angle
a = zeros(num_steps, 1);     % Acceleration

% =========================================================================
% CLOSED-LOOP ALGORITHM PLACEHOLDER
% Loop through your time steps, read the corresponding sensor data,
% fuse the inputs, and calculate the new steering angle and acceleration.
% =========================================================================
for i = 1:num_steps
    current_time = t(i);
    
    % --- INSERT YOUR FRIEND'S SENSOR FUSION & CONTROL ALGORITHM HERE ---
    
    % Example Dummy Logic (matching your requested behavior):
    % High acceleration to go fast
    a(i) = 30.0; 
    
    % Sharp left turn at 2.0 seconds
    if current_time >= 2.0
        delta(i) = 0.5; % 0.5 radians (left turn)
    else
        delta(i) = 0.0;
    end
end
% =========================================================================

%% 3. Create TimeSeries Structures for Simulink
disp('Creating MATLAB timeseries objects for Simulink...');

% Create Timeseries for Steering Angle
steering_ts = timeseries(delta, t, 'Name', 'SteeringAngle');
steering_ts.DataInfo.Units = 'radians';

% Create Timeseries for Acceleration
accel_ts = timeseries(a, t, 'Name', 'Acceleration');
accel_ts.DataInfo.Units = 'm/s^2';

% Group them into a structure for easy loading
ego_controls = struct();
ego_controls.SteeringAngle = steering_ts;
ego_controls.Acceleration = accel_ts;

%% 4. Save to Output .mat File
output_filename = 'ego_controls_simulink.mat';
save(output_filename, 'ego_controls', 'steering_ts', 'accel_ts');

disp(['Success! Controls saved to: ', output_filename]);
disp('You can now use a "From Workspace" or "From File" block in Simulink to read these timeseries objects.');
