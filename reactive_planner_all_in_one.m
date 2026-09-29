% =========================================================================
% ALL-IN-ONE AUTONOMOUS VEHICLE REACTIVE PLANNER SIMULATION SCRIPT
% =========================================================================
% THREE CORE PILLARS INCLUDED:
%   i)  KALMAN FILTERING   : State tracking & relative velocity estimation
%   ii) RISK SCORING       : Continuous spatio-temporal threat (TTC + Gaussian)
%   iii) LATERAL OFFSET    : Dynamic waypoint shift & smooth steering evasion
% =========================================================================

clc; clear; close all;

disp('=================================================================');
disp('   AUTONOMOUS VEHICLE REACTIVE PLANNER SIMULATION (ALL-IN-ONE)');
disp('   Features: Kalman Filter | Risk Scoring | Dynamic Lateral Offset');
disp('=================================================================');

%% 1. SIMULATION PARAMETERS & SETUP
dt = 0.1;           % Time step (seconds)
sim_time = 12.0;    % Total simulation duration (seconds)
t = 0:dt:sim_time;
N = length(t);

% Ego Vehicle Physical & Kinematic Parameters
L_wheelbase = 2.7;  % Wheelbase (meters)
v_ego = 5.0;        % Initial longitudinal speed (m/s) (~18 km/h)

% Ego Vehicle Pose [x (m), y (m), yaw (rad)]
ego_x   = 0.0;
ego_y   = 0.0;
ego_yaw = 0.0;

% Target Goal Position
goal_position = [50.0, 0.0];

% Reference Route Waypoints (Center Lane Corridor)
waypoints = [ 0.0,  0.0;
             10.0,  0.0;
             20.0,  0.0;
             30.0,  0.0;
             40.0,  0.0;
             50.0,  0.0];

% Stationary/Slow Obstacle Position (X = 25m, Y = 0.1m)
obs_true_x = 25.0;
obs_true_y = 0.1;

%% 2. DATA LOGGING ARRAYS
log_ego_x          = zeros(N, 1);
log_ego_y          = zeros(N, 1);
log_ego_yaw        = zeros(N, 1);
log_raw_meas_x     = zeros(N, 1);
log_raw_meas_y     = zeros(N, 1);
log_kf_x           = zeros(N, 1);
log_kf_y           = zeros(N, 1);
log_risk_score     = zeros(N, 1);
log_lateral_offset = zeros(N, 1);
log_steering       = zeros(N, 1);
log_throttle       = zeros(N, 1);
log_brake          = zeros(N, 1);

%% 3. CLOSED-LOOP SIMULATION EXECUTION LOOP
for k = 1:N
    % Record current Ego pose
    log_ego_x(k)   = ego_x;
    log_ego_y(k)   = ego_y;
    log_ego_yaw(k) = ego_yaw;
    
    % --- Sensor Detection Simulation (with Noise & Range Limits) ---
    rel_true_x = obs_true_x - ego_x;
    rel_true_y = obs_true_y - ego_y;
    
    % Transform global relative position into ego body relative frame
    rel_body_x =  cos(ego_yaw)*rel_true_x + sin(ego_yaw)*rel_true_y;
    rel_body_y = -sin(ego_yaw)*rel_true_x + cos(ego_yaw)*rel_true_y;
    
    % Inject Sensor Noise (std = 0.25m) within sensor range horizon (35m)
    if rel_body_x > 0 && rel_body_x < 35.0
        noise_x = 0.25 * randn();
        noise_y = 0.20 * randn();
        raw_x = rel_body_x + noise_x;
        raw_y = rel_body_y + noise_y;
    else
        % No detection / Out of sensor range
        raw_x = 0;
        raw_y = 0;
    end
    
    log_raw_meas_x(k) = raw_x;
    log_raw_meas_y(k) = raw_y;
    
    % Pack fused tracks array [track_id, class_id, rel_x, rel_y]
    fused_tracks = [1, 1, raw_x, raw_y];
    
    % --- Call Embedded Reactive Planner Local Function ---
    [steering, throttle, brake, obs_status, closest_x, risk_score, lateral_offset] = ...
        reactive_planner_core(fused_tracks, ego_x, ego_y, ego_yaw, goal_position, waypoints, dt);
    
    % Log States & Controls
    log_risk_score(k)     = risk_score;
    log_lateral_offset(k) = lateral_offset;
    log_steering(k)       = steering;
    log_throttle(k)       = throttle;
    log_brake(k)          = brake;
    log_kf_x(k)           = closest_x;
    
    % --- Longitudinal Speed & Kinematic Bicycle Model Motion Update ---
    accel = 1.5 * throttle - 3.0 * brake;
    v_ego = max(0.5, v_ego + accel * dt);
    
    % Kinematic Bicycle Update
    ego_x   = ego_x   + v_ego * cos(ego_yaw) * dt;
    ego_y   = ego_y   + v_ego * sin(ego_yaw) * dt;
    ego_yaw = ego_yaw + (v_ego / L_wheelbase) * tan(steering) * dt;
    
    % Output Periodic Diagnostics to Command Window
    if mod(k, 10) == 0 || k == 1
        fprintf('Time: %4.1fs | Ego X: %5.2fm, Y: %5.2fm | Risk: %4.2f | Offset: %5.2fm | Steer: %5.2frad | Brake: %4.2f\n', ...
            t(k), ego_x, ego_y, risk_score, lateral_offset, steering, brake);
    end
end

disp('=================================================================');
disp('   SIMULATION COMPLETE. GENERATING VISUAL DIAGNOSTICS PLOT...');
disp('=================================================================');

%% 4. DIAGNOSTIC GRAPHICAL VISUALIZATION
figure('Name', 'Reactive Planner (Kalman, Risk Score, Lateral Offset)', ...
       'Position', [100, 100, 1100, 800], 'Color', 'w');

% --- Subplot 1: 2D Trajectory & Dynamic Obstacle Evasion ---
subplot(2, 2, 1);
plot(waypoints(:,1), waypoints(:,2), 'k--', 'LineWidth', 1.5, 'DisplayName', 'Nominal Waypoints');
hold on;
plot(log_ego_x, log_ego_y, 'b-', 'LineWidth', 2.0, 'DisplayName', 'Ego Trajectory');
plot(obs_true_x, obs_true_y, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'DisplayName', 'Obstacle');
xlabel('X Position (m)'); ylabel('Y Position (m)');
title('\bf 1. Vehicle Evasion Trajectory');
legend('Location', 'best'); grid on; axis equal;

% --- Subplot 2: Dynamic Spatio-Temporal Risk Score ---
subplot(2, 2, 2);
plot(t, log_risk_score, 'r-', 'LineWidth', 2.0);
hold on;
yline(0.35, 'm--', 'Hazard Threshold (0.35)', 'LineWidth', 1.2);
yline(0.70, 'k--', 'Braking Threshold (0.70)', 'LineWidth', 1.2);
xlabel('Time (s)'); ylabel('Risk Score [0 - 1]');
title('\bf 2. Continuous Risk Score R(t)');
grid on; ylim([0, 1.05]);

% --- Subplot 3: Raw Sensor Detections vs Kalman Filter Estimation ---
subplot(2, 2, 3);
valid_idx = log_raw_meas_x > 0;
plot(t(valid_idx), log_raw_meas_x(valid_idx), 'r.', 'MarkerSize', 8, 'DisplayName', 'Noisy Sensor Raw');
hold on;
plot(t, log_kf_x, 'g-', 'LineWidth', 2.0, 'DisplayName', 'Kalman Filter Track');
xlabel('Time (s)'); ylabel('Longitudinal Distance (m)');
title('\bf 3. Kalman Filter Noise Reduction');
legend('Location', 'best'); grid on;

% --- Subplot 4: Dynamic Lateral Offset & Steering Output ---
subplot(2, 2, 4);
yyaxis left;
plot(t, log_lateral_offset, 'b-', 'LineWidth', 2.0);
ylabel('Lateral Offset (m)', 'Color', 'b');
yyaxis right;
plot(t, log_steering, 'm-', 'LineWidth', 1.5);
ylabel('Steering Angle (rad)', 'Color', 'm');
xlabel('Time (s)');
title('\bf 4. Dynamic Lateral Offset & Steering Command');
grid on;

disp('All plots rendered successfully.');


% =========================================================================
% LOCAL FUNCTION: REACTIVE PLANNER CORE ALGORITHM
% =========================================================================
function [steering, throttle, brake, obstacle_status, closest_obstacle_x, risk_score, lateral_offset] = ...
    reactive_planner_core(fused_tracks, ego_x, ego_y, ego_yaw, goal_position, waypoints, dt)

% Default Controls & Outputs
steering           = 0;
throttle           = 0.4;
brake              = 0;
obstacle_status    = 0;
closest_obstacle_x = 999;
risk_score         = 0.0;
lateral_offset     = 0.0;

% Persistent State for Kalman Filter & Smooth Lateral Trajectory Offset
persistent x_est P_est current_offset

if isempty(x_est)
    x_est = [999; 0; 0; 0]; % [pos_x; pos_y; vel_x; vel_y]
end
if isempty(P_est)
    P_est = eye(4) * 1.0;
end
if isempty(current_offset)
    current_offset = 0.0;
end

% -------------------------------------------------------------------------
% i) KALMAN FILTERING (Track State & Relative Velocity Estimation)
% -------------------------------------------------------------------------
% Constant Velocity State Transition Matrix A
A_kf = [1  0  dt  0;
        0  1  0  dt;
        0  0  1   0;
        0  0  0   1];

% Measurement Observation Matrix H
H_kf = [1 0 0 0;
        0 1 0 0];

Q_kf = eye(4) * 0.05; % Process Noise Covariance
R_kf = eye(2) * 0.25; % Measurement Noise Covariance

% Kalman Predict Step
x_pred = A_kf * x_est;
P_pred = A_kf * P_est * A_kf' + Q_kf;

% Extract raw measurement track
raw_meas_x = 999;
raw_meas_y = 0;
min_raw_dist = 999;

if size(fused_tracks, 2) >= 4 && ~isempty(fused_tracks)
    for i = 1:size(fused_tracks, 1)
        px = fused_tracks(i, 3);
        py = fused_tracks(i, 4);
        if px == 0 && py == 0, continue; end
        dist = sqrt(px^2 + py^2);
        if dist < min_raw_dist && px > -2.0
            min_raw_dist = dist;
            raw_meas_x = px;
            raw_meas_y = py;
        end
    end
end

% Kalman Update Step
if raw_meas_x < 150
    z = [raw_meas_x; raw_meas_y];
    K_gain = P_pred * H_kf' / (H_kf * P_pred * H_kf' + R_kf);
    x_est  = x_pred + K_gain * (z - H_kf * x_pred);
    P_est  = (eye(4) - K_gain * H_kf) * P_pred;
else
    x_est = x_pred;
    P_est = P_pred;
end

target_x  = x_est(1); % Filtered Longitudinal Distance (m)
target_y  = x_est(2); % Filtered Lateral Displacement (m)
rel_vel_x = x_est(3); % Filtered Relative Velocity (m/s)

closest_obstacle_x = target_x;

% -------------------------------------------------------------------------
% ii) RISK SCORING MODEL (Spatio-Temporal Threat)
% -------------------------------------------------------------------------
closing_speed = -rel_vel_x;
if closing_speed > 0.1 && target_x > 0
    ttc = target_x / closing_speed;
else
    ttc = target_x / max(0.5, 4.0);
end

% Spatial Gaussian Decay Risk Factor
sigma_x = 10.0;
sigma_y = 1.4;
spatial_risk = exp(-0.5 * (target_x / sigma_x)^2) * exp(-0.5 * (target_y / sigma_y)^2);

% Time-To-Collision (TTC) Risk Factor
ttc_tau = 3.0;
if ttc < 10.0 && target_x > 0
    ttc_risk = exp(-ttc / ttc_tau);
else
    ttc_risk = 0.0;
end

% Combined Risk Score [0.0 - 1.0]
risk_score = min(1.0, max(0.0, 0.6 * spatial_risk + 0.4 * ttc_risk));

if risk_score > 0.35
    obstacle_status = 1;
end

% -------------------------------------------------------------------------
% iii) DYNAMIC LATERAL OFFSET TRAJECTORY PLANNING
% -------------------------------------------------------------------------
LANE_CLEARANCE = 1.8; % meters to shift
target_lateral_offset = 0.0;

if obstacle_status == 1 && target_x > 0 && target_x < 15.0
    if target_y >= 0
        evade_dir = -1.0; % Shift right
    else
        evade_dir = 1.0;  % Shift left
    end
    target_lateral_offset = evade_dir * LANE_CLEARANCE * risk_score;
end

% Smooth offset transition low-pass filter
alpha_offset   = 0.15;
current_offset = (1 - alpha_offset) * current_offset + alpha_offset * target_lateral_offset;
lateral_offset = current_offset;

% -------------------------------------------------------------------------
% WAYPOINT STEERING & SPEED CONTROL
% -------------------------------------------------------------------------
num_waypoints = size(waypoints, 1);
if num_waypoints < 2, return; end

% Find closest waypoint
closest_wp  = 1;
min_wp_dist = 1e9;
for i = 1:num_waypoints
    dx_wp = waypoints(i,1) - ego_x;
    dy_wp = waypoints(i,2) - ego_y;
    d_wp  = sqrt(dx_wp^2 + dy_wp^2);
    if d_wp < min_wp_dist
        min_wp_dist = d_wp;
        closest_wp  = i;
    end
end

target_wp = min(num_waypoints, closest_wp + 1);
wp_x = waypoints(target_wp, 1);
wp_y = waypoints(target_wp, 2);

dx_route  = wp_x - ego_x;
dy_route  = wp_y - ego_y;
route_yaw = atan2(dy_route, dx_route);

% Project lateral offset perpendicular to route heading
offset_wp_x = wp_x - lateral_offset * sin(route_yaw);
offset_wp_y = wp_y + lateral_offset * cos(route_yaw);

dx_offset = offset_wp_x - ego_x;
dy_offset = offset_wp_y - ego_y;
desired_heading = atan2(dy_offset, dx_offset);

heading_error = desired_heading - ego_yaw;
while heading_error > pi,  heading_error = heading_error - 2*pi; end
while heading_error < -pi, heading_error = heading_error + 2*pi; end

STEERING_GAIN = 0.75;
MAX_STEERING  = 0.45;
steering = STEERING_GAIN * heading_error;
steering = max(-MAX_STEERING, min(MAX_STEERING, steering));

% Speed & Braking Modulation
goal_x    = goal_position(1);
goal_y    = goal_position(2);
goal_dist = sqrt((goal_x - ego_x)^2 + (goal_y - ego_y)^2);

if goal_dist < 2.0
    steering = 0; throttle = 0; brake = 0.8; return;
end

if risk_score > 0.70
    throttle = 0.0;
    brake    = min(0.9, 0.4 + 0.5 * risk_score);
elseif risk_score > 0.35
    throttle = max(0.1, 0.4 * (1.0 - risk_score));
    brake    = 0.0;
else
    throttle = 0.4;
    brake    = 0.0;
end

end
