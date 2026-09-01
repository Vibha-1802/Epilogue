% export_radar_to_csv.m
% Extracts radar data from r_data.mat and saves it to a clean CSV
% so it can be parsed by Python for the Risk Scoring Algorithm.

clc; clear; close all;

%% 1. Load Data
disp('Loading r_data.mat...');
if ~isfile('r_data.mat')
    error('r_data.mat not found in current directory.');
end

% Load the data
data = load('r_data.mat');

% Find the variable that holds the Detections.
% Usually, it's called 'Detections' or it's the only variable in the file.
varNames = fieldnames(data);
if isfield(data, 'Detections')
    dets = data.Detections;
else
    dets = data.(varNames{1});
end

%% 2. Extract Fields and Write to CSV
disp('Extracting fields...');

% Open file
fid = fopen('radar_detections.csv', 'w');
% Write Header
fprintf(fid, 'Time,ObjectID,X,Y,VX,VY\n');

num_dets = length(dets);

for i = 1:num_dets
    det = dets{i}; % If it's a cell array. If object array, use dets(i)
    if isobject(dets) || isstruct(dets)
        det = dets(i);
    end
    
    % Time
    t = det.Time;
    
    % Object ID (TargetIndex)
    obj_id = det.ObjectAttributes{1}.TargetIndex;
    if isempty(obj_id)
        obj_id = -1;
    end
    
    % Measurement (Assuming standard [x; y; vx; vy; ...] format)
    meas = det.Measurement;
    x = meas(1);
    y = meas(2);
    vx = meas(3);
    vy = meas(4);
    
    % Write to CSV
    fprintf(fid, '%f,%d,%f,%f,%f,%f\n', t, obj_id, x, y, vx, vy);
end

fclose(fid);
disp('Success! Data exported to radar_detections.csv');
