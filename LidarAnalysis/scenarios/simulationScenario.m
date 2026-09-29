function [allData, scnro, sensor] = generateSensorData()
%GENERATESENSORDATA Generate sensor data from the driving scenario.
%
% The ego vehicle controls the end of the simulation.
% Non-ego actors stop at their final waypoint and remain there.
%
% Existing scenario geometry, actor parameters, trajectories, and
% lidar configuration are preserved.

    %% Create driving scenario
    [scnro, egoVehicle] = createDrivingScenario;

    %% Create sensor
    sensor = createSensor(scnro);

    %% Register sensor with ego vehicle
    addSensors(scnro, sensor, egoVehicle.ActorID);

    %% Initialize output data structure
    allData = struct( ...
        'Time', {}, ...
        'ActorPoses', {}, ...
        'ObjectDetections', {}, ...
        'LaneDetections', {}, ...
        'PointClouds', {}, ...
        'INSMeasurements', {});

    %% Run scenario
    running = true;

    while running

        % Current simulation time
        time = scnro.SimulationTime;

        % Get target poses when the sensor supports this operation
        if ~isa(sensor, 'insSensor')
            poses = targetPoses(scnro, sensor.SensorIndex);
        end

        % Initialize sensor outputs
        laneDetections = [];
        objectDetections = [];
        insMeas = [];

        % Generate lidar point cloud
        [ptClouds, isValidPointCloudTime] = sensor();

        % Store sensor data when a valid point-cloud timestamp occurs
        if isValidPointCloudTime

            allData(end + 1) = struct( ...
                'Time', scnro.SimulationTime, ...
                'ActorPoses', actorPoses(scnro), ...
                'ObjectDetections', {objectDetections}, ...
                'LaneDetections', {laneDetections}, ...
                'PointClouds', {ptClouds}, ...
                'INSMeasurements', {insMeas});

        end

        % Advance the scenario by one simulation step.
        %
        % The simulation will continue while:
        %   1. The ego vehicle has not completed its trajectory.
        %   2. StopTime has not been reached.
        %
        % Non-ego actors have sufficiently long final waits so they
        % remain stationary instead of ending the scenario early.
        running = advance(scnro);

    end

    %% Reset scenario and release sensor
    restart(scnro);
    release(sensor);

end


%% ========================================================================
%  CREATE SENSOR
%  ========================================================================

function sensor = createSensor(scnro)

    % Obtain actor profiles from the scenario
    profiles = actorProfiles(scnro);

    % Lidar configuration
    %
    % These parameters are kept unchanged from the original generated code.
    sensor = lidarPointCloudGenerator( ...
        SensorIndex = 1, ...
        SensorLocation = [1.5 0], ...
        HasEgoVehicle = false, ...
        RangeAccuracy = 0.02, ...
        ElevationResolution = 0.4, ...
        ActorProfiles = profiles);

end


%% ========================================================================
%  CREATE DRIVING SCENARIO
%  ========================================================================

function [scnro, egoVehicle] = createDrivingScenario()

    %% --------------------------------------------------------------------
    % Create scenario
    % ---------------------------------------------------------------------

    scnro = drivingScenario;


    %% --------------------------------------------------------------------
    % Roads
    % ---------------------------------------------------------------------

    roadCenters = [ ...
        29.83   34.6   0;
        29.73  -40     0];

    road(scnro, roadCenters, ...
        Name = 'Road');


    roadCenters = [ ...
        49.7  -10.2   0;
         9.8  -10.4   0];

    road2 = road(scnro, roadCenters, ...
        Name = 'Road1');


    %% --------------------------------------------------------------------
    % Guardrail
    % ---------------------------------------------------------------------

    barrier( ...
        scnro, ...
        road2, ...
        RoadEdge = 'right', ...
        ClassID = 6, ...
        Width = 0.433, ...
        Mesh = driving.scenario.guardrailMesh, ...
        PlotColor = [0.55 0.55 0.55], ...
        Name = 'Guardrail');


    %% --------------------------------------------------------------------
    % Jersey barrier
    % ---------------------------------------------------------------------

    barrierCenters = [ ...
        30.1  -37.8   0;
        28.1  -38.0   0];

    barrier( ...
        scnro, ...
        barrierCenters, ...
        ClassID = 5, ...
        Width = 0.61, ...
        Height = 0.81, ...
        Mesh = driving.scenario.jerseyBarrierMesh, ...
        PlotColor = [0.65 0.65 0.65], ...
        Name = 'Jersey Barrier');


    %% ====================================================================
    % EGO VEHICLE
    % ====================================================================

    egoVehicle = vehicle( ...
        scnro, ...
        ClassID = 1, ...
        Position = [31.7 31.6 0], ...
        Mesh = driving.scenario.carMesh, ...
        Name = 'Car');


    % Original ego waypoints
    waypoints = [ ...
        31.7   31.6   0;
        31.3  -33.8   0];


    % Original ego speed
    speed = [30; 30];


    % IMPORTANT:
    % Ego trajectory is unchanged.
    %
    % No wait is added here.
    % Therefore, when the ego reaches its final waypoint,
    % it can terminate the scenario.
    trajectory( ...
        egoVehicle, ...
        waypoints, ...
        speed);


    %% ====================================================================
    % NON-EGO CAR
    % ====================================================================

    car1 = vehicle( ...
        scnro, ...
        ClassID = 1, ...
        Position = [28.6 3 0], ...
        Mesh = driving.scenario.carMesh, ...
        Name = 'Car1');


    % Original car waypoints
    waypoints = [ ...
        28.6   3.0    0;
        28.6  29.4    0];


    % Original speed is maintained.
    %
    % At the final waypoint:
    %   speed = 0
    %   waittime = 60 seconds
    %
    % This makes the car stop at its final point and remain there.
    speed = [15; 0];

    waittime = [0; 60];

    trajectory( ...
        car1, ...
        waypoints, ...
        speed, ...
        waittime);


    %% ====================================================================
    % PEDESTRIAN
    % ====================================================================

    pedestrian = actor( ...
        scnro, ...
        ClassID = 4, ...
        Length = 0.24, ...
        Width = 0.45, ...
        Height = 1.7, ...
        Position = [27.8 0.4 0], ...
        RCSPattern = [-8 -8; -8 -8], ...
        Mesh = driving.scenario.pedestrianMesh, ...
        Name = 'Pedestrian');


    % Original pedestrian waypoints
    waypoints = [ ...
        27.80    0.40    0;
        27.76   -7.15    0.01];


    % Original speed is maintained.
    % Final speed is zero so the pedestrian remains stationary.
    speed = [1.5; 0];

    waittime = [0; 60];

    trajectory( ...
        pedestrian, ...
        waypoints, ...
        speed, ...
        waittime);


    %% ====================================================================
    % TRUCK
    % ====================================================================

    truck = vehicle( ...
        scnro, ...
        ClassID = 2, ...
        Length = 8.2, ...
        Width = 2.5, ...
        Height = 3.5, ...
        Position = [44.9 -10.8 0], ...
        RearOverhang = 1, ...
        FrontOverhang = 0.9, ...
        Mesh = driving.scenario.truckMesh, ...
        Name = 'Truck');


    % Original truck waypoints
    waypoints = [ ...
        44.9   -10.8    0;
        21.63  -10.83   0.01];


    % Original truck speed is maintained.
    % Truck stops at the second waypoint.
    speed = [20; 0];

    waittime = [0; 60];

    trajectory( ...
        truck, ...
        waypoints, ...
        speed, ...
        waittime);


    %% ====================================================================
    % BICYCLE
    % ====================================================================

    bicycle = actor( ...
        scnro, ...
        ClassID = 3, ...
        Length = 1.7, ...
        Width = 0.45, ...
        Height = 1.7, ...
        Position = [32.06 -15.22 0.01], ...
        Mesh = driving.scenario.bicycleMesh, ...
        Name = 'Bicycle');


    % Original bicycle waypoints
    waypoints = [ ...
        32.06  -15.22   0.01;
        28.10  -19.70   0;
        28.20  -31.30   0;
        28.20  -33.60   0];


    % Original bicycle speed is maintained.
    % Bicycle stops at the final waypoint.
    speed = [5; 5; 5; 0];

    waittime = [0; 0; 0; 60];

    trajectory( ...
        bicycle, ...
        waypoints, ...
        speed, ...
        waittime);

end