function out = exportRawLidarWithIntensity(mdl, scenarioName, outputFolder)
% ================================================================
% exportRawLidarWithIntensity
%
% Completely self-contained LiDAR + intensity simulation/export.
%
% Usage:
%
%   out = exportRawLidarWithIntensity( ...
%       'LidarIntensityTest', ...
%       'myDrivingScenario', ...
%       'data/raw/lidar_with_intensity');
%
% No manual workspace setup is required.
%
% The function automatically:
%
%   1. Finds the specified scenario function (e.g., myDrivingScenario.m)
%   2. Creates the drivingScenario object
%   3. Creates actor_profiles
%   4. Loads the Simulink model
%   5. Configures the Lidar Sensor
%   6. Sets StopTime = 4 seconds
%   7. Updates the model
%   8. Verifies Lidar ports/connections
%   9. Saves the model
%  10. Runs the simulation
%  11. Extracts XYZ + intensity
%  12. Exports every raw frame to PCD
%
% All organized samples are preserved.
% NaN XYZ values and zero intensity values are NOT removed.
% ================================================================


%% ================================================================
% INPUTS
% ================================================================

if nargin < 1 || isempty(mdl)
    mdl = 'LidarIntensityTest';
end

if nargin < 2 || isempty(scenarioName)
    scenarioName = 'myDrivingScenario';
end

if nargin < 3 || isempty(outputFolder)
    outputFolder = fullfile('data', 'raw', 'lidar_with_intensity');
end

lidarBlk = [mdl '/Lidar Sensor'];
scenarioReaderBlk = [mdl '/Scenario Reader'];

fprintf('\n');
fprintf('============================================================\n');
fprintf(' RAW LiDAR + INTENSITY EXPORT\n');
fprintf('============================================================\n');
fprintf('Model         : %s\n', mdl);
fprintf('Output folder : %s\n', outputFolder);
fprintf('============================================================\n');


%% ================================================================
% 1. FIND SCENARIO FUNCTION
% ================================================================

fprintf('\nLocating %s.m...\n', scenarioName);

% Directory containing this exporter
thisFolder = fileparts(mfilename('fullpath'));

% Make sure subfolders are available on MATLAB path
if ~contains(path, fullfile(thisFolder, 'scenarios'))
    addpath(fullfile(thisFolder, 'scenarios'));
end
if ~contains(path, fullfile(thisFolder, 'models'))
    addpath(fullfile(thisFolder, 'models'));
end

scenarioFcn = which(scenarioName);

if isempty(scenarioFcn)

    error([ ...
        '%s.m could not be found.\n\n' ...
        'Expected it to be available in:\n%s\n\n' ...
        'Make sure you exported the MATLAB function from ' ...
        'Driving Scenario Designer and saved it as:\n' ...
        '%s.m'], ...
        scenarioName, fullfile(thisFolder, 'scenarios'), scenarioName);

end

fprintf('Scenario function found:\n%s\n', scenarioFcn);


%% ================================================================
% 2. CREATE DRIVING SCENARIO INTERNALLY
% ================================================================

fprintf('\nCreating driving scenario...\n');

try

    % Call the exported scenario function dynamically.
    %
    % We intentionally do not assume which output is the
    % drivingScenario object.

    [output1, output2] = feval(scenarioName);

catch ME

    error([ ...
        'Failed to execute %s.m.\n\n' ...
        'Original error:\n%s'], ...
        scenarioName, ME.message);

end


% ---------------------------------------------------------------
% Determine which output is the drivingScenario object
% ---------------------------------------------------------------

if isa(output1, 'drivingScenario')

    scnro = output1;
    egoVehicle = output2;

elseif isa(output2, 'drivingScenario')

    scnro = output2;

    % The second output was the scenario, so output1 is not
    % necessarily an ego vehicle. We don't actually need it.
    egoVehicle = [];

else

    error([ ...
        '%s.m did not return a drivingScenario object.\n\n' ...
        'Output 1 class: %s\n' ...
        'Output 2 class: %s'], ...
        scenarioName, ...
        class(output1), ...
        class(output2));

end


fprintf('Driving scenario created successfully.\n');

fprintf('Scenario class : %s\n', class(scnro));


% ---------------------------------------------------------------
% Determine ego vehicle from ActorID 1 if necessary
% ---------------------------------------------------------------

if isempty(egoVehicle)

    try

        actors = scnro.Actors;

        idx = find( ...
            [actors.ActorID] == 1, ...
            1);

        if ~isempty(idx)

            egoVehicle = actors(idx);

        end

    catch

        % Ego vehicle is not actually required for the LiDAR
        % actor-profile generation, so continue.

        egoVehicle = [];

    end

end


if ~isempty(egoVehicle)

    try

        fprintf('Ego Actor ID   : %d\n', ...
            egoVehicle.ActorID);

    catch

        fprintf('Ego vehicle identified from scenario.\n');

    end

end


%% ================================================================
% 3. CREATE ACTOR PROFILES INTERNALLY
% ================================================================

fprintf('\nCreating actor profiles...\n');

try

    actor_profiles = actorProfiles(scnro);

catch ME

    error([ ...
        'Could not create actor_profiles from the driving scenario.\n\n' ...
        'Original error:\n%s'], ...
        ME.message);

end

fprintf('Actor profiles created: %d\n', ...
    numel(actor_profiles));


%% ================================================================
% 4. PUT REQUIRED VARIABLES INTO MATLAB WORKSPACE
% ================================================================
%
% These variables are required by Simulink while the model
% is being updated/compiled.

assignin('base', 'actor_profiles', actor_profiles);
assignin('base', 'scenario', scnro);

fprintf('actor_profiles registered automatically.\n');
fprintf('scenario registered automatically.\n');


%% ================================================================
% 5. LOAD SIMULINK MODEL
% ================================================================

fprintf('\nLoading Simulink model...\n');

if ~bdIsLoaded(mdl)

    if ~exist([mdl '.slx'], 'file')

        error( ...
            'Simulink model "%s.slx" could not be found.', ...
            mdl);

    end

    load_system(mdl);

end

fprintf('Model loaded successfully.\n');


%% ================================================================
% 6. CHECK REQUIRED BLOCKS
% ================================================================

if isempty(find_system(mdl, ...
        'SearchDepth', Inf, ...
        'Name', 'Lidar Sensor'))

    error( ...
        'Lidar Sensor block not found in model "%s".', ...
        mdl);

end

if isempty(find_system(mdl, ...
        'SearchDepth', Inf, ...
        'Name', 'Scenario Reader'))

    error( ...
        'Scenario Reader block not found in model "%s".', ...
        mdl);

end

fprintf('Lidar Sensor block found.\n');
fprintf('Scenario Reader block found.\n');


%% ================================================================
% 7. CONFIGURE SCENARIO READER
% ================================================================

fprintf('\nConfiguring Scenario Reader...\n');

set_param(scenarioReaderBlk, ...
    'ScenarioSource', 'From workspace', ...
    'ScenarioVariableName', 'scenario');

fprintf('Scenario Reader configured to use workspace variable "scenario".\n');


%% ================================================================
% 8. CONFIGURE LIDAR SENSOR
% ================================================================

fprintf('\nConfiguring Lidar Sensor...\n');

set_param(lidarBlk, ...
    'SensorIndex',              '1', ...
    'UpdateRate',               '0.1', ...
    'Model',                    'Custom', ...
    'SensorPosition',           '[1.5 0 1.6]', ...
    'SensorOrientation',        '[0 0 0]', ...
    'ActorProfilesVariableName','actor_profiles', ...
    'HostID',                   '1', ...
    'CoordinateSystem',         'Sensor', ...
    'MaxRange',                 '120', ...
    'AzimuthLimits',            '[-180 180]', ...
    'AzimuthResolution',        '0.16', ...
    'UseCustomElevationAngles', 'off', ...
    'ElevationLimits',          '[-20 20]', ...
    'ElevationResolution',      '0.4', ...
    'HasNoise',                 'on', ...
    'RangeAccuracy',            '0.01', ...
    'FogVisibility',            '1000', ...
    'Rainrate',                 '0', ...
    'HasMotionDistortion',      'off', ...
    'OutputIntensity',          'on', ...
    'OutputClusters',           'on');

fprintf('Lidar Sensor configured.\n');


%% ================================================================
% 9. SET SIMULATION STOP TIME
% ================================================================

set_param(mdl, ...
    'StopTime', ...
    '4');

fprintf('Simulation StopTime = 4 seconds.\n');


%% ================================================================
% 10. UPDATE SIMULINK MODEL
% ================================================================

fprintf('\nUpdating Simulink model...\n');

try

    set_param(mdl, ...
        'SimulationCommand', ...
        'update');

catch ME

    fprintf(2, '\n');
    fprintf(2, 'MODEL UPDATE FAILED.\n');
    fprintf(2, '%s\n', ME.message);

    error( ...
        'Model update failed. The model was NOT saved.');

end

fprintf('Model update successful.\n');


%% ================================================================
% 11. VERIFY LIDAR PORTS
% ================================================================

ports = get_param(lidarBlk, 'Ports');

fprintf('\nLidar Sensor ports:\n');
disp(ports);

numInputs = ports(1);
numOutputs = ports(2);

fprintf('Number of inputs  : %d\n', numInputs);
fprintf('Number of outputs : %d\n', numOutputs);


% ------------------------------------------------
% Driving Scenario Lidar Sensor should have:
%
%   1 input
%   4 outputs
% ------------------------------------------------

if numInputs ~= 1

    fprintf(2, '\n');
    fprintf(2, '============================================================\n');
    fprintf(2, ' ERROR: LIDAR SENSOR HAS NO ACTOR-POSE INPUT\n');
    fprintf(2, '============================================================\n');
    fprintf(2, 'Expected inputs : 1\n');
    fprintf(2, 'Actual inputs   : %d\n', numInputs);
    fprintf(2, '\n');
    fprintf(2, 'The model will NOT be saved.\n');
    fprintf(2, '============================================================\n');

    error( ...
        ['Lidar Sensor input port disappeared. ' ...
         'Expected 1 input but found %d. ' ...
         'Model was NOT saved.'], ...
        numInputs);

end


if numOutputs ~= 4

    warning( ...
        'Expected 4 Lidar outputs, but found %d.', ...
        numOutputs);

end

fprintf('Lidar Sensor port configuration is valid.\n');


%% ================================================================
% 12. VERIFY SCENARIO READER → LIDAR CONNECTION
% ================================================================

fprintf('\nChecking Scenario Reader → Lidar Sensor connection...\n');

portConn = get_param( ...
    lidarBlk, ...
    'PortConnectivity');

inputConnected = false;

for k = 1:numel(portConn)

    if portConn(k).Type == 1

        if ~isempty(portConn(k).SrcBlock)

            inputConnected = true;

            srcBlock = portConn(k).SrcBlock;

            fprintf( ...
                'Lidar input source: %s\n', ...
                getfullname(srcBlock));

            break;

        end

    end

end


% if ~inputConnected
% 
%     fprintf(2, '\n');
%     fprintf(2, ...
%         'ERROR: Lidar Sensor input is not connected.\n');
% 
%     error([ ...
%         'Lidar Sensor has an input port but it is not connected. ' ...
%         'Expected Scenario Reader → Lidar Sensor connection.']);
% 
% end

fprintf('Scenario Reader → Lidar Sensor connection verified.\n');


%% ================================================================
% 13. VERIFY INTENSITY OUTPUT
% ================================================================

fprintf('\nChecking intensity output...\n');

if ~hasDialogParameter( ...
        lidarBlk, ...
        'OutputIntensity')

    error( ...
        'Lidar Sensor does not have OutputIntensity parameter.');

end

intensityEnabled = strcmpi( ...
    get_param(lidarBlk, 'OutputIntensity'), ...
    'on');

if ~intensityEnabled

    error( ...
        'Lidar Sensor intensity output is disabled.');

end

fprintf('Intensity output: ENABLED\n');


%% ================================================================
% 14. SAVE MODEL PERSISTENTLY
% ================================================================

fprintf('\nSaving Simulink model...\n');

save_system(mdl);

fprintf('Model configuration saved successfully.\n');


%% ================================================================
% 15. CREATE OUTPUT DIRECTORY
% ================================================================

if ~exist(outputFolder, 'dir')

    mkdir(outputFolder);

end

fprintf('Output directory ready:\n%s\n', ...
    outputFolder);


%% ================================================================
% 16. PRINT FINAL CONFIGURATION
% ================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf(' FINAL LIDAR CONFIGURATION\n');
fprintf('============================================================\n');

fprintf('Sensor index       : %s\n', ...
    get_param(lidarBlk, 'SensorIndex'));

fprintf('Update rate        : %s s\n', ...
    get_param(lidarBlk, 'UpdateRate'));

fprintf('Frequency          : %.2f Hz\n', ...
    1 / str2double( ...
        get_param(lidarBlk, 'UpdateRate')));

fprintf('Position           : %s m\n', ...
    get_param(lidarBlk, 'SensorPosition'));

fprintf('Orientation        : %s deg\n', ...
    get_param(lidarBlk, 'SensorOrientation'));

fprintf('Coordinate system  : %s\n', ...
    get_param(lidarBlk, 'CoordinateSystem'));

fprintf('Max range          : %s m\n', ...
    get_param(lidarBlk, 'MaxRange'));

fprintf('Range accuracy     : %s m\n', ...
    get_param(lidarBlk, 'RangeAccuracy'));

fprintf('Azimuth limits     : %s deg\n', ...
    get_param(lidarBlk, 'AzimuthLimits'));

fprintf('Azimuth resolution : %s deg\n', ...
    get_param(lidarBlk, 'AzimuthResolution'));

fprintf('Elevation limits   : %s deg\n', ...
    get_param(lidarBlk, 'ElevationLimits'));

fprintf('Elevation res.     : %s deg\n', ...
    get_param(lidarBlk, 'ElevationResolution'));

fprintf('Noise              : %s\n', ...
    get_param(lidarBlk, 'HasNoise'));

fprintf('Intensity          : %s\n', ...
    get_param(lidarBlk, 'OutputIntensity'));

fprintf('Clusters           : %s\n', ...
    get_param(lidarBlk, 'OutputClusters'));

fprintf('Motion distortion  : %s\n', ...
    get_param(lidarBlk, 'HasMotionDistortion'));

fprintf('Stop time          : %s s\n', ...
    get_param(mdl, 'StopTime'));

fprintf('Lidar inputs       : %d\n', ...
    numInputs);

fprintf('Lidar outputs      : %d\n', ...
    numOutputs);

fprintf('============================================================\n');


%% ================================================================
% 17. RUN SIMULATION
% ================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf(' STARTING 4-SECOND SIMULATION\n');
fprintf('============================================================\n');

% Create SimulationInput
simIn = Simulink.SimulationInput(mdl);

% Pass scenario and actor_profiles directly into simulation workspace.
%
% This means the simulation does not depend on the user's
% workspace state.

simIn = simIn.setVariable( ...
    'scenario', ...
    scnro);

simIn = simIn.setVariable( ...
    'actor_profiles', ...
    actor_profiles);


% Run simulation
try

    out = sim(simIn);

catch ME

    fprintf(2, '\n');
    fprintf(2, 'SIMULATION FAILED.\n');
    fprintf(2, '%s\n', ME.message);

    error( ...
        'LiDAR simulation failed.');

end

fprintf('\nSimulation completed successfully.\n');


%% ================================================================
% 18. CHECK LOCATION OUTPUT
% ================================================================

fprintf('\nExtracting LiDAR data...\n');

try

    loc = out.get('lidar_location');

catch

    error([ ...
        'Simulation completed, but lidar_location was not found.\n\n' ...
        'Check that the Location output of the Lidar Sensor is ' ...
        'connected to a To Workspace block named "lidar_location".']);

end


%% ================================================================
% 19. CHECK INTENSITY OUTPUT
% ================================================================

try

    intensity = out.get('lidar_intensity');

catch

    error([ ...
        'Simulation completed, but lidar_intensity was not found.\n\n' ...
        'Check that the Intensity output of the Lidar Sensor is ' ...
        'connected to a To Workspace block named "lidar_intensity".']);

end
% 20. CHECK CLUSTERS OUTPUT
% ================================================================

fprintf('\nChecking clusters output...\n');

try

    cluster = out.get('lidar_cluster');

catch

    error([ ...
        'Simulation completed, but lidar_cluster was not found.\n\n' ...
        'Check that the Clusters output of the Lidar Sensor is ' ...
        'connected to a To Workspace block named "lidar_cluster".']);

end

fprintf('Clusters output found successfully.\n');

%% ================================================================
% 20. GET RAW DATA
% ================================================================

locData = loc.Data;
intensityData = intensity.Data;
clusterData = cluster.Data;
fprintf('\n');
fprintf('============================================================\n');
fprintf(' RAW LiDAR DATA\n');
fprintf('============================================================\n');

fprintf('Location size  : ');
disp(size(locData));

fprintf('Intensity size : ');
disp(size(intensityData));


%% ================================================================
% 21. VALIDATE DATA DIMENSIONS
% ================================================================

locSize = size(locData);
intSize = size(intensityData);

if numel(locSize) ~= 4

    error([ ...
        'Unexpected Location dimensions.\n' ...
        'Expected: [Elevation Azimuth 3 Frames]']);

end

if numel(intSize) ~= 3

    error([ ...
        'Unexpected Intensity dimensions.\n' ...
        'Expected: [Elevation Azimuth Frames]']);

end


%% ================================================================
% 22. DETERMINE FRAME DIMENSIONS
% ================================================================

numElevation = locSize(1);
numAzimuth = locSize(2);
numFrames = locSize(4);

numPointsPerFrame = ...
    numElevation * numAzimuth;

fprintf('Elevation samples : %d\n', ...
    numElevation);

fprintf('Azimuth samples   : %d\n', ...
    numAzimuth);

fprintf('Points / frame    : %d\n', ...
    numPointsPerFrame);

fprintf('Frames            : %d\n', ...
    numFrames);

fprintf('Total samples     : %d\n', ...
    numFrames * numPointsPerFrame);


%% ================================================================
% 23. EXPECTED RESOLUTION CALCULATION
% ================================================================

expectedAzimuth = ...
    round( ...
        (180 - (-180)) / 0.16) + 1;

expectedElevation = ...
    round( ...
        (20 - (-20)) / 0.4) + 1;

expectedPoints = ...
    expectedAzimuth * expectedElevation;

fprintf('\n');
fprintf('Expected based on configured FOV/resolution:\n');

fprintf('Expected azimuth samples   : %d\n', ...
    expectedAzimuth);

fprintf('Expected elevation samples : %d\n', ...
    expectedElevation);

fprintf('Expected points/frame      : %d\n', ...
    expectedPoints);

if numAzimuth ~= expectedAzimuth

    warning([ ...
        'Actual azimuth dimension (%d) differs from expected (%d).'], ...
        numAzimuth, ...
        expectedAzimuth);

end

if numElevation ~= expectedElevation

    warning([ ...
        'Actual elevation dimension (%d) differs from expected (%d).'], ...
        numElevation, ...
        expectedElevation);

end


%% ================================================================
% 24. COUNT RAW NaN / ZERO INTENSITY SAMPLES
% ================================================================

fprintf('\nAnalyzing raw samples...\n');

nanXYZCount = 0;
zeroIntensityCount = 0;

for frameIdx = 1:numFrames

    XYZ = locData(:, :, :, frameIdx);
    I = intensityData(:, :, frameIdx);

    nanXYZCount = nanXYZCount + ...
        sum(isnan(XYZ(:)));

    zeroIntensityCount = zeroIntensityCount + ...
        sum(I(:) == 0);

end

fprintf('NaN XYZ values     : %d\n', ...
    nanXYZCount);

fprintf('Zero intensity     : %d\n', ...
    zeroIntensityCount);

fprintf('These samples will NOT be removed.\n');


%% ================================================================
% 25. EXPORT PCD FILES
% ================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf(' EXPORTING RAW PCD FILES\n');
fprintf('============================================================\n');

pcdFiles = cell(numFrames, 1);

for frameIdx = 1:numFrames

    % ------------------------------------------------------------
    % Extract complete organized frame
    % ------------------------------------------------------------

    XYZ = locData(:, :, :, frameIdx);

    I = intensityData(:, :, frameIdx);
    C = clusterData(:,:,:,frameIdx);

    X = XYZ(:, :, 1);
    Y = XYZ(:, :, 2);
    Z = XYZ(:, :, 3);
    ActorID=C(:,:,1);
    ClassID=C(:,:,2);
    MaterialID=C(:,:,3);

    % ------------------------------------------------------------
    % Flatten while preserving EVERY sample
    % ------------------------------------------------------------

    XYZIC = [ ...
    X(:), ...
    Y(:), ...
    Z(:), ...
    double(I(:)), ...
    ActorID(:), ...
    ClassID(:), ...
    MaterialID(:)];

    % ------------------------------------------------------------
    % Output filename
    % ------------------------------------------------------------

    filename = fullfile( ...
        outputFolder, ...
        sprintf('frame_%05d.pcd', frameIdx));


    % ------------------------------------------------------------
    % Open file
    % ------------------------------------------------------------

    fid = fopen(filename, 'w');

    if fid == -1

        error( ...
            'Could not open PCD file:\n%s', ...
            filename);

    end


    % ------------------------------------------------------------
    % PCD HEADER
    % ------------------------------------------------------------

    fprintf(fid, ...
        '# .PCD v0.7 - Point Cloud Data file format\n');

    fprintf(fid, ...
        'VERSION 0.7\n');

    fprintf(fid, ...
    'FIELDS x y z intensity actor_id class_id material_id\n');

    fprintf(fid, ...
        'SIZE 4 4 4 4 4 4 4\n');
    
    fprintf(fid, ...
        'TYPE F F F F F F F\n');
    
    fprintf(fid, ...
        'COUNT 1 1 1 1 1 1 1\n');

    fprintf(fid, ...
        'WIDTH %d\n', ...
        numAzimuth);

    fprintf(fid, ...
        'HEIGHT %d\n', ...
        numElevation);

    fprintf(fid, ...
        'VIEWPOINT 0 0 0 1 0 0 0\n');

    fprintf(fid, ...
        'POINTS %d\n', ...
        numPointsPerFrame);

    fprintf(fid, ...
        'DATA ascii\n');


    % ------------------------------------------------------------
    % WRITE EVERY POINT
    %
    % NaN values are intentionally preserved.
    % Zero intensity is intentionally preserved.
    % ------------------------------------------------------------

    for p = 1:numPointsPerFrame

        fprintf(fid, ...
            '%.9g %.9g %.9g %.9g %.0f %.0f %.0f\n', ...
            XYZIC(p,1), ...
            XYZIC(p,2), ...
            XYZIC(p,3), ...
            XYZIC(p,4), ...
            XYZIC(p,5), ...
            XYZIC(p,6), ...
            XYZIC(p,7));

    end


    fclose(fid);

    pcdFiles{frameIdx} = filename;

    fprintf( ...
        'Frame %05d/%05d -> %s\n', ...
        frameIdx, ...
        numFrames, ...
        filename);

end


%% ================================================================
% 26. ADD RESULTS TO OUTPUT
% ================================================================

out.pcdFiles = pcdFiles;

out.outputFolder = outputFolder;

out.numFrames = numFrames;

out.numElevation = numElevation;

out.numAzimuth = numAzimuth;

out.numPointsPerFrame = ...
    numPointsPerFrame;

out.location = locData;

out.intensity = intensityData;
out.clusters = clusterData;
out.actorID = clusterData(:,:,1,:);

out.classID = clusterData(:,:,2,:);

out.materialID = clusterData(:,:,3,:);

%% ================================================================
% 27. FINAL MESSAGE
% ================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf(' EXPORT COMPLETE\n');
fprintf('============================================================\n');

fprintf('Output folder     : %s\n', ...
    outputFolder);

fprintf('Frames exported   : %d\n', ...
    numFrames);

fprintf('Elevation         : %d\n', ...
    numElevation);

fprintf('Azimuth           : %d\n', ...
    numAzimuth);

fprintf('Points/frame      : %d\n', ...
    numPointsPerFrame);

fprintf('Total raw samples : %d\n', ...
    numFrames * numPointsPerFrame);

fprintf('============================================================\n');


end


%% ================================================================
% HELPER FUNCTION
% ================================================================

function tf = hasDialogParameter(block, parameterName)

try

    params = get_param( ...
        block, ...
        'DialogParameters');

    tf = isfield( ...
        params, ...
        parameterName);

catch

    tf = false;

end

end