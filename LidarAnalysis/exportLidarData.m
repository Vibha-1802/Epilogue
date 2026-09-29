%% Export LiDAR data from allData to individual PCD files

clear;
clc;

%% Generate LiDAR data
[allData, scnro, sensor] = simulationScenario();

%% Create output directory
outputFolder = fullfile(pwd, 'lidar_data');

if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

%% Number of valid LiDAR frames
numFrames = length(allData);

fprintf('Number of LiDAR frames: %d\n', numFrames);
fprintf('Output folder: %s\n\n', outputFolder);

%% Export every LiDAR frame
for k = 1:numFrames

    % Get simulation time
    t = allData(k).Time;

    % Get point cloud
    ptCloud = allData(k).PointClouds;

    % File name
    filename = fullfile( ...
        outputFolder, ...
        sprintf('frame_%06d_t_%07.3f.pcd', k, t));

    % Write PCD
    pcwrite( ...
        ptCloud, ...
        filename, ...
        Encoding = "binary");

    fprintf( ...
        'Frame %4d / %4d | Time = %7.3f s | Points = %7d\n', ...
        k, ...
        numFrames, ...
        t, ...
        ptCloud.Count);
end

fprintf('\nFinished exporting LiDAR data.\n');
fprintf('Files saved in:\n%s\n', outputFolder);