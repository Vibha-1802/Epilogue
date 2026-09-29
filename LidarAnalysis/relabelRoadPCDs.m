function info = relabelRoadPCDs(pcdFolder, scenarioName, varargin)
% ================================================================
% relabelRoadPCDs
%
% Adds a semantic ROAD / TERRAIN label to the static ground points
% of LiDAR frames exported by exportRawLidarWithIntensity.
%
% WHY THIS IS NEEDED
% ------------------
% The Driving Scenario Designer "Class Editor" only defines ACTOR
% classes (Car, Truck, Bicycle, Pedestrian, Jersey Barrier,
% Guardrail). Roads are not actors, so a road can never be given a
% ClassID there. The Lidar Sensor block does return the road/ground
% surface, but it reports every ground hit under ONE reserved
% surface ID - it does not tell you whether that hit was on the
% drivable road or on off-road terrain.
%
% METHOD
% ------
% For every exported frame this function:
%
%   1. Replays the scenario to the timestamp of that frame.
%   2. Asks the scenario for the road boundaries expressed in EGO
%      coordinates  ->  roadBoundaries(egoVehicle)
%   3. Transforms the LiDAR points from SENSOR coordinates into EGO
%      coordinates using the sensor mounting pose.
%   4. Runs a point-in-polygon test of each GROUND CANDIDATE point
%      against the road-surface polygons.
%         inside  any road polygon -> RoadClassID
%         outside every road polygon -> TerrainClassID
%
% GROUND CANDIDATES ARE DELIBERATELY RESTRICTIVE
% ----------------------------------------------
% A point is only allowed to become Road or Terrain when ALL of the
% following hold:
%
%   * its XYZ is finite (it is a real return, not an empty ray)
%   * its ActorID is NOT the ActorID of any scenario entity - this
%     covers actors AND barrier segments, so cars, trucks,
%     pedestrians, bicycles, jersey barriers and guardrails are
%     never overwritten
%   * it is NOT an unlabelled point (ActorID == 0 AND ClassID == 0)
%
% The last rule is the important one: unlabelled points keep
% ClassID 0 and are never promoted to Road, because an unlabelled
% point is not guaranteed to be ground.
%
% USAGE
% -----
%   info = relabelRoadPCDs( ...
%       'data/raw/road_label_test', ...
%       'road_label_test');
%
%   info = relabelRoadPCDs( ...
%       'data/raw/road_label_test', ...
%       'road_label_test', ...
%       'OutputFolder', 'data/labeled/road_label_test', ...
%       'Model',        'road_label_test_model');
% ================================================================


%% ================================================================
% 0. PARSE INPUTS
% ================================================================

p = inputParser;

addRequired(p, 'pcdFolder',    @(x) ischar(x) || isstring(x));
addRequired(p, 'scenarioName', @(x) ischar(x) || isstring(x));

addParameter(p, 'OutputFolder',      '',           @(x) ischar(x) || isstring(x));
addParameter(p, 'Model',             '',           @(x) ischar(x) || isstring(x));
addParameter(p, 'SensorPosition',    [1.5 0 1.6],  @(x) isnumeric(x) && numel(x) == 3);
addParameter(p, 'SensorOrientation', [0 0 0],      @(x) isnumeric(x) && numel(x) == 3);
addParameter(p, 'UpdateRate',        0.1,          @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'RoadClassID',       7,            @(x) isnumeric(x) && isscalar(x));
addParameter(p, 'TerrainClassID',    8,            @(x) isnumeric(x) && isscalar(x));
addParameter(p, 'EdgeTolerance',     0.05,         @(x) isnumeric(x) && isscalar(x) && x >= 0);

parse(p, pcdFolder, scenarioName, varargin{:});

opts = p.Results;

pcdFolder    = char(opts.pcdFolder);
scenarioName = char(opts.scenarioName);
mdl          = char(opts.Model);

outputFolder = char(opts.OutputFolder);

if isempty(outputFolder)
    outputFolder = [regexprep(pcdFolder, '[\\/]+$', '') '_labeled'];
end

fprintf('\n');
fprintf('============================================================\n');
fprintf(' ROAD / TERRAIN SEMANTIC RELABELLING\n');
fprintf('============================================================\n');
fprintf('Input frames  : %s\n', pcdFolder);
fprintf('Output frames : %s\n', outputFolder);
fprintf('Scenario      : %s\n', scenarioName);
fprintf('============================================================\n');


%% ================================================================
% 1. MAKE PROJECT SUBFOLDERS VISIBLE
% ================================================================

thisFolder = fileparts(mfilename('fullpath'));

addpath(fullfile(thisFolder, 'scenarios'));
addpath(fullfile(thisFolder, 'models'));


%% ================================================================
% 2. OPTIONALLY READ THE REAL SENSOR MOUNTING FROM THE MODEL
% ================================================================
%
% The lidar points are stored in SENSOR coordinates, so the sensor
% mounting pose is what converts them back into EGO coordinates.
% Reading it from the model avoids a silent mismatch if the block
% was retuned after this function was written.

sensorPosition    = opts.SensorPosition(:).';
sensorOrientation = opts.SensorOrientation(:).';
updateRate        = opts.UpdateRate;

if ~isempty(mdl)

    fprintf('\nReading Lidar Sensor mounting from model "%s"...\n', mdl);

    wasLoaded = bdIsLoaded(mdl);

    if ~wasLoaded
        load_system(mdl);
    end

    lidarBlk = find_system(mdl, ...
        'SearchDepth',  Inf, ...
        'BlockType',    'MATLABSystem', ...
        'Name',         'Lidar Sensor');

    if isempty(lidarBlk)
        lidarBlk = find_system(mdl, 'SearchDepth', Inf, 'Name', 'Lidar Sensor');
    end

    if isempty(lidarBlk)

        warning('Lidar Sensor block not found in "%s". Using supplied mounting pose.', mdl);

    else

        blk = lidarBlk{1};

        sensorPosition    = str2num(get_param(blk, 'SensorPosition'));    %#ok<ST2NM>
        sensorOrientation = str2num(get_param(blk, 'SensorOrientation')); %#ok<ST2NM>
        updateRate        = str2double(get_param(blk, 'UpdateRate'));

        coordSys = get_param(blk, 'CoordinateSystem');

        fprintf('Coordinate system  : %s\n', coordSys);

        if ~contains(lower(coordSys), 'sensor')

            % Points are already in ego coordinates, so no sensor
            % offset must be applied.

            warning(['Lidar CoordinateSystem is "%s", not Sensor. ' ...
                     'Treating exported points as EGO coordinates.'], coordSys);

            sensorPosition    = [0 0 0];
            sensorOrientation = [0 0 0];

        end

    end

    if ~wasLoaded
        close_system(mdl, 0);
    end

end

fprintf('\nSensor position    : [%g %g %g] m\n', sensorPosition);
fprintf('Sensor orientation : [%g %g %g] deg\n', sensorOrientation);
fprintf('Update rate        : %g s\n', updateRate);


%% ================================================================
% 3. BUILD THE SCENARIO AND LOCK IT TO THE LIDAR FRAME RATE
% ================================================================

fprintf('\nCreating driving scenario...\n');

[scnro, egoVehicle] = feval(scenarioName);

if ~isa(scnro, 'drivingScenario')
    error('%s did not return a drivingScenario object.', scenarioName);
end

if isempty(egoVehicle)

    actors = scnro.Actors;
    idx    = find([actors.ActorID] == 1, 1);

    if isempty(idx)
        error('Could not determine the ego vehicle from "%s".', scenarioName);
    end

    egoVehicle = actors(idx);

end

% Stepping the scenario at exactly the lidar update rate means
% scenario step k lines up with exported frame k, with frame 1 at
% t = 0.
%
% StopTime must be set explicitly. Without it, advance() halts as
% soon as the FIRST actor finishes its trajectory, which happens
% long before the Simulink run ended. With StopTime set, actors
% that finish early simply hold their final pose - exactly what
% the Scenario Reader block does.

scnro.SampleTime = updateRate;

restart(scnro);

% Every entity the lidar can label, including barrier segments.
%
% scnro.Actors alone is NOT enough: barriers (guardrails, jersey
% barriers) are not actors, so their ActorIDs (10001+, 20001+)
% would otherwise be mistaken for ground and overwritten.
% actorProfiles returns exactly the id set the Lidar Sensor block
% itself is given, so it covers actors and barriers alike.

entityIDs = unique([ ...
    reshape([scnro.Actors.ActorID],       1, []), ...
    reshape([actorProfiles(scnro).ActorID], 1, [])]);

fprintf('Ego ActorID        : %d\n', egoVehicle.ActorID);
fprintf('Scenario entities  : %s\n', mat2str(entityIDs));


%% ================================================================
% 4. COLLECT THE EXPORTED FRAMES
% ================================================================

files = dir(fullfile(pcdFolder, '*.pcd'));

if isempty(files)
    error('No .pcd files found in:\n%s', pcdFolder);
end

[~, order] = sort({files.name});
files      = files(order);

numFrames = numel(files);

fprintf('Frames found       : %d\n', numFrames);

% Frame k is sampled at t = (k-1)*updateRate, so the replay has to
% survive at least that long.

scnro.StopTime = (numFrames + 1) * updateRate;

if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end


%% ================================================================
% 5. ROTATION FROM SENSOR FRAME TO EGO FRAME
% ================================================================

R = eulerToRotation(sensorOrientation);
t = sensorPosition;


%% ================================================================
% 6. RELABEL EVERY FRAME
% ================================================================

fprintf('\n');
fprintf('------------------------------------------------------------\n');
fprintf(' frame        ground        road     terrain   off-road %%\n');
fprintf('------------------------------------------------------------\n');

totalGround  = 0;
totalRoad    = 0;
totalTerrain = 0;
totalRescued = 0;

groundClassesSeen = [];

outFiles  = cell(numFrames, 1);
frameStats = zeros(numFrames, 3);

scenarioAlive = true;

for k = 1:numFrames

    inFile = fullfile(files(k).folder, files(k).name);

    [header, data] = readPcdAscii(inFile);

    xyz      = data(:, 1:3);
    actorID  = data(:, 5);
    classID  = data(:, 6);

    % ------------------------------------------------------------
    % 6a. Ground candidate mask
    %
    %     Real return  AND  not a scenario actor  AND  not an
    %     unlabelled point.
    % ------------------------------------------------------------

    isFinitePt = all(isfinite(xyz), 2);

    isEntityPt = ismember(actorID, entityIDs);

    isUnlabelled = (actorID == 0) & (classID == 0);

    isGround = isFinitePt & ~isEntityPt & ~isUnlabelled;

    groundClassesSeen = union(groundClassesSeen, unique(classID(isGround)));

    % ------------------------------------------------------------
    % 6b. Road polygons in EGO coordinates at this timestamp
    % ------------------------------------------------------------

    rbEgo = roadBoundaries(egoVehicle);

    % ------------------------------------------------------------
    % 6c. Sensor coordinates -> ego coordinates
    % ------------------------------------------------------------

    ptsEgo = xyz(isGround, :) * R.' + t;

    % ------------------------------------------------------------
    % 6d. Point in road polygon test
    % ------------------------------------------------------------

    [onRoad, onRoadStrict] = pointsOnRoad( ...
        ptsEgo(:, 1), ptsEgo(:, 2), rbEgo, opts.EdgeTolerance);

    % ------------------------------------------------------------
    % 6e. Write the semantic labels back
    % ------------------------------------------------------------

    groundIdx = find(isGround);

    classID(groundIdx( onRoad)) = opts.RoadClassID;
    classID(groundIdx(~onRoad)) = opts.TerrainClassID;

    data(:, 6) = classID;

    nGround  = numel(groundIdx);
    nRoad    = sum(onRoad);
    nTerrain = nGround - nRoad;
    nRescued = sum(onRoad & ~onRoadStrict);

    totalGround  = totalGround  + nGround;
    totalRoad    = totalRoad    + nRoad;
    totalTerrain = totalTerrain + nTerrain;
    totalRescued = totalRescued + nRescued;

    frameStats(k, :) = [nGround nRoad nTerrain];

    if nGround > 0
        offRoadPct = 100 * nTerrain / nGround;
    else
        offRoadPct = 0;
    end

    fprintf('%6d  %12d  %10d  %10d  %9.2f\n', ...
        k, nGround, nRoad, nTerrain, offRoadPct);

    % ------------------------------------------------------------
    % 6f. Save
    % ------------------------------------------------------------

    outFile = fullfile(outputFolder, files(k).name);

    writePcdAscii(outFile, header, data);

    outFiles{k} = outFile;

    % ------------------------------------------------------------
    % 6g. Step the scenario to the next lidar timestamp
    % ------------------------------------------------------------

    if k < numFrames && scenarioAlive

        scenarioAlive = advance(scnro);

        if ~scenarioAlive

            % The ego has reached the end of its trajectory. Every
            % later frame was rendered from this same frozen pose,
            % so holding it is the correct behaviour.

            fprintf('  (scenario frozen from frame %d onwards)\n', k + 1);

        end

    end

end

fprintf('------------------------------------------------------------\n');


%% ================================================================
% 7. WRITE THE CLASS MAP NEXT TO THE DATA
% ================================================================
%
% Road and Terrain cannot live in the Class Editor, so the full
% label map is written beside the frames instead.

classMap = buildClassMap(scnro, opts.RoadClassID, opts.TerrainClassID);

mapFile = fullfile(outputFolder, 'class_map.json');

fid = fopen(mapFile, 'w');
fprintf(fid, '%s', jsonencode(classMap, 'PrettyPrint', true));
fclose(fid);

fprintf('\nClass map written : %s\n', mapFile);


%% ================================================================
% 8. SUMMARY
% ================================================================

if totalGround > 0
    offRoadPctTotal = 100 * totalTerrain / totalGround;
else
    offRoadPctTotal = 0;
end

fprintf('\n');
fprintf('============================================================\n');
fprintf(' RELABELLING COMPLETE\n');
fprintf('============================================================\n');
fprintf('Frames processed        : %d\n', numFrames);
fprintf('Ground candidate points : %d\n', totalGround);
fprintf('  -> Road    (ClassID %d): %d\n', opts.RoadClassID, totalRoad);
fprintf('  -> Terrain (ClassID %d): %d\n', opts.TerrainClassID, totalTerrain);
fprintf('Off-road share          : %.2f %%\n', offRoadPctTotal);
fprintf('Edge tolerance          : %.3f m (rescued %d points)\n', ...
    opts.EdgeTolerance, totalRescued);
fprintf('Original ClassIDs on ground candidates: %s\n', mat2str(groundClassesSeen(:).'));
fprintf('============================================================\n');

info = struct( ...
    'outputFolder',   outputFolder, ...
    'files',          {outFiles}, ...
    'numFrames',      numFrames, ...
    'frameStats',     frameStats, ...
    'totalGround',    totalGround, ...
    'totalRoad',      totalRoad, ...
    'totalTerrain',   totalTerrain, ...
    'totalEdgeRescued', totalRescued, ...
    'classMap',       classMap, ...
    'sensorPosition', sensorPosition, ...
    'roadClassID',    opts.RoadClassID, ...
    'terrainClassID', opts.TerrainClassID);

end


%% ================================================================
% HELPER: point in any road polygon
% ================================================================

function [tf, tfStrict] = pointsOnRoad(x, y, rbEgo, tol)

tf       = false(numel(x), 1);
tfStrict = tf;

if isempty(x)
    return;
end

if ~iscell(rbEgo)
    rbEgo = {rbEgo};
end

for i = 1:numel(rbEgo)

    b = rbEgo{i};

    if isempty(b) || size(b, 2) < 2
        continue;
    end

    bx = b(:, 1);
    by = b(:, 2);

    [in, on] = inpolygon(x, y, bx, by);

    tf = tf | in | on;

end

tfStrict = tf;

% ------------------------------------------------------------
% Edge tolerance band.
%
% A return that lands within EdgeTolerance of a road edge is
% still counted as road. The Lidar Sensor applies range noise
% (RangeAccuracy, 1 cm by default), so rays aimed at the very
% last centimetres of the road surface land just outside the
% road polygon. Without this band those points are reported as
% Terrain even though no terrain exists.
%
% 5 cm is far below any meaningful road/terrain separation, so
% it absorbs the noise without swallowing real off-road ground.
% Set EdgeTolerance to 0 to disable it.
% ------------------------------------------------------------

if tol > 0

    remaining = find(~tf);

    if ~isempty(remaining)

        for i = 1:numel(rbEgo)

            b = rbEgo{i};

            if isempty(b) || size(b, 2) < 2
                continue;
            end

            d = pointToPolylineDistance( ...
                x(remaining), y(remaining), b(:, 1), b(:, 2));

            near = d <= tol;

            tf(remaining(near)) = true;

            remaining = remaining(~near);

            if isempty(remaining)
                break;
            end

        end

    end

end

end


%% ================================================================
% HELPER: shortest distance from points to a polyline
% ================================================================

function d = pointToPolylineDistance(px, py, bx, by)

valid = isfinite(bx) & isfinite(by);

bx = bx(valid);
by = by(valid);

d = inf(numel(px), 1);

for s = 1:numel(bx) - 1

    ax = bx(s);      ay = by(s);
    ex = bx(s + 1);  ey = by(s + 1);

    dx = ex - ax;
    dy = ey - ay;

    len2 = dx * dx + dy * dy;

    if len2 == 0
        seg = hypot(px - ax, py - ay);
    else
        u   = ((px - ax) * dx + (py - ay) * dy) / len2;
        u   = min(max(u, 0), 1);
        seg = hypot(px - (ax + u * dx), py - (ay + u * dy));
    end

    d = min(d, seg);

end

end


%% ================================================================
% HELPER: yaw/pitch/roll (degrees) -> rotation matrix
% ================================================================

function R = eulerToRotation(ypr)

yaw   = deg2rad(ypr(1));
pitch = deg2rad(ypr(2));
roll  = deg2rad(ypr(3));

Rz = [cos(yaw) -sin(yaw) 0; sin(yaw) cos(yaw) 0; 0 0 1];
Ry = [cos(pitch) 0 sin(pitch); 0 1 0; -sin(pitch) 0 cos(pitch)];
Rx = [1 0 0; 0 cos(roll) -sin(roll); 0 sin(roll) cos(roll)];

R = Rz * Ry * Rx;

end


%% ================================================================
% HELPER: read an ASCII PCD written by exportRawLidarWithIntensity
% ================================================================

function [header, data] = readPcdAscii(file)

fid = fopen(file, 'r');

if fid == -1
    error('Could not open PCD file:\n%s', file);
end

header = {};

while true

    line = fgetl(fid);

    if ~ischar(line)
        fclose(fid);
        error('Reached end of file before DATA line in:\n%s', file);
    end

    header{end + 1} = line; %#ok<AGROW>

    if strncmpi(strtrim(line), 'DATA', 4)
        break;
    end

end

numFields = numel(strsplit(strtrim(regexprep(header{ ...
    find(strncmpi(header, 'FIELDS', 6), 1)}, '^FIELDS\s*', ''))));

raw = textscan(fid, repmat('%f', 1, numFields), ...
    'CollectOutput', true);

fclose(fid);

data = raw{1};

end


%% ================================================================
% HELPER: write an ASCII PCD with the original header
% ================================================================

function writePcdAscii(file, header, data)

fid = fopen(file, 'w');

if fid == -1
    error('Could not open PCD file for writing:\n%s', file);
end

for i = 1:numel(header)
    fprintf(fid, '%s\n', header{i});
end

% Vectorised write. The transpose feeds the format string column
% by column, which is far faster than one fprintf per point.

fprintf(fid, '%.9g %.9g %.9g %.9g %.0f %.0f %.0f\n', data.');

fclose(fid);

end


%% ================================================================
% HELPER: assemble the full label map
% ================================================================

function classMap = buildClassMap(scnro, roadClassID, terrainClassID)

entries = struct('class_id', {}, 'name', {}, 'kind', {});

% ------------------------------------------------------------
% Every class the Lidar Sensor can emit, taken from the same
% actorProfiles list the sensor itself is given. This covers
% actors and barriers, so the map always matches the Class
% Editor contents that are actually used in the scenario.
% ------------------------------------------------------------

profiles = actorProfiles(scnro);

actors   = scnro.Actors;
actorIDs = [actors.ActorID];

barriers = scnro.Barriers;

seen = [];

for i = 1:numel(profiles)

    cid = profiles(i).ClassID;

    if ismember(cid, seen)
        continue;
    end

    seen(end + 1) = cid; %#ok<AGROW>

    aid = profiles(i).ActorID;

    idx = find(actorIDs == aid, 1);

    if ~isempty(idx)

        name = actors(idx).Name;
        kind = 'actor';

    else

        % Barrier segment ids are numbered 10000*group + segment,
        % so the leading digits identify which barrier it is.

        grp = floor(double(aid) / 10000);

        if grp >= 1 && grp <= numel(barriers)
            name = barriers(grp).Name;
        else
            name = sprintf('Barrier %d', cid);
        end

        kind = 'barrier';

    end

    entries(end + 1) = struct( ...
        'class_id', cid, ...
        'name',     name, ...
        'kind',     kind); %#ok<AGROW>

end

% ------------------------------------------------------------
% Static surface classes added by this function. These cannot
% exist in the Class Editor because roads are not actors.
% ------------------------------------------------------------

entries(end + 1) = struct( ...
    'class_id', roadClassID, ...
    'name',     'Road', ...
    'kind',     'static_surface');

entries(end + 1) = struct( ...
    'class_id', terrainClassID, ...
    'name',     'Terrain', ...
    'kind',     'static_surface');

entries(end + 1) = struct( ...
    'class_id', 0, ...
    'name',     'Unlabelled', ...
    'kind',     'unlabelled');

[~, order] = sort([entries.class_id]);

entries = entries(order);

classMap = struct( ...
    'description', ['ClassID map for exported LiDAR frames. ' ...
                    'Road and Terrain are assigned geometrically by ' ...
                    'relabelRoadPCDs.m, not by the Class Editor.'], ...
    'classes',     entries);

end
