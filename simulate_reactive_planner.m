% =========================================================================
% SIMULATION TEST SCRIPT FOR ADVANCED REACTIVE PLANNER
% Demonstrates:
%   1. Kalman Filter tracking of noisy obstacle detections
%   2. Dynamic Spatio-Temporal Risk Scoring (TTC + Gaussian spatial threat)
%   3. Continuous Lateral Offset path generation & smooth collision avoidance
% =========================================================================

clc; clear; close all;
clear reactive_planner; % Clear persistent state inside the planner function

disp('=================================================================');
disp('   AUTONOMOUS VEHICLE REACTIVE PLANNER SIMULATION');
disp('   Features: Kalman Filter | Risk Score | Dynamic Lateral Offset');
disp('=================================================================');

%% 1. Simulation Parameters & Setup
dt = 0.1;           % Time step (seconds)
sim_time = 12.0;    % Total simulation duration (seconds)
t = 0:dt:sim_time;
N = length(t);

% Vehicle Kinematic Constants
L_wheelbase = 2.7;  % Wheelbase (m)
v_ego = 5.0;        % Initial cruise speed (m/s) (~18 km/h)

% Ego Vehicle Initial Pose [x, y, yaw]
ego_x   = 0.0;
ego_y   = 0.0;
ego_yaw = 0.0;

% Goal Position
goal_position = [50.0, 0.0];

% Reference Waypoints (Straight Lane Corridor)
waypoints = [ 0.0,  0.0;
             10.0,  0.0;
             20.0,  0.0;
             30.0,  0.0;
             40.0,  0.0;
             50.0,  0.0];

% Simulated Obstacle Setup (Stationary vehicle in center lane at X = 25m, Y = 0.1m)
obs_true_x = 25.0;
obs_true_y = 0.1;

%% 2. Data Logging Initialization
log_ego_x         = zeros(N, 1);
log_ego_y         = zeros(N, 1);
log_ego_yaw       = zeros(N, 1);
log_raw_meas_x    = zeros(N, 1);
log_raw_meas_y    = zeros(N, 1);
log_kf_x          = zeros(N, 1);
log_kf_y          = zeros(N, 1);
log_risk_score    = zeros(N, 1);
log_lateral_offset= zeros(N, 1);
log_steering      = zeros(N, 1);
log_throttle      = zeros(N, 1);
log_brake         = zeros(N, 1);

%% 3. Closed-Loop Simulation Loop
for k = 1:N
    % Current Ego Pose
    log_ego_x(k)   = ego_x;
    log_ego_y(k)   = ego_y;
    log_ego_yaw(k) = ego_yaw;
    
    % --- Sensor Measurement Simulation (with Gaussian Noise & Dropouts) ---
    rel_true_x = obs_true_x - ego_x;
    rel_true_y = obs_true_y - ego_y;
    
    % Transform global relative offset to ego body frame
    rel_body_x =  cos(ego_yaw)*rel_true_x + sin(ego_yaw)*rel_true_y;
    rel_body_y = -sin(ego_yaw)*rel_true_x + cos(ego_yaw)*rel_true_y;
    
    % Add sensor measurement noise (std = 0.25m)
    if rel_body_x > 0 && rel_body_x < 35.0
        noise_x = 0.25 * randn();
        noise_y = 0.20 * randn();
        raw_x = rel_body_x + noise_x;
        raw_y = rel_body_y + noise_y;
    else
        % Out of sensor range
        raw_x = 0;
        raw_y = 0;
    end
    
    log_raw_meas_x(k) = raw_x;
    log_raw_meas_y(k) = raw_y;
    
    % Pack fused tracks array [track_id, class_id, rel_x, rel_y]
    fused_tracks = [1, 1, raw_x, raw_y];
    
    % --- Call Reactive Planner Algorithm ---
    [steering, throttle, brake, obs_status, closest_x, risk_score, lateral_offset] = ...
        reactive_planner(fused_tracks, ego_x, ego_y, ego_yaw, goal_position, waypoints);
    
    % Log Controller Outputs & Planner States
    log_risk_score(k)     = risk_score;
    log_lateral_offset(k) = lateral_offset;
    log_steering(k)       = steering;
    log_throttle(k)       = throttle;
    log_brake(k)          = brake;
    log_kf_x(k)           = closest_x;
    
    % --- Update Vehicle Longitudinal Speed & Kinematic Bicycle Model ---
    accel = 1.5 * throttle - 3.0 * brake;
    v_ego = max(0.5, v_ego + accel * dt); % Update velocity
    
    % Kinematic Bicycle Update
    ego_x   = ego_x   + v_ego * cos(ego_yaw) * dt;
    ego_y   = ego_y   + v_ego * sin(ego_yaw) * dt;
    ego_yaw = ego_yaw + (v_ego / L_wheelbase) * tan(steering) * dt;
    
    % Console Diagnostics Output
    if mod(k, 10) == 0 || k == 1
        fprintf('Time: %4.1fs | Ego X: %5.2fm, Y: %5.2fm | Risk: %4.2f | Offset: %5.2fm | Steer: %5.2frad | Brake: %4.2f\n', ...
            t(k), ego_x, ego_y, risk_score, lateral_offset, steering, brake);
    end
end

disp('=================================================================');
disp('   SIMULATION COMPLETE. GENERATING DIAGNOSTIC PLOTS...');
disp('=================================================================');

%% 4. Graphical Visualization & Diagnostics
figure('Name', 'Reactive Planner Diagnostics (Kalman, Risk Score, Lateral Offset)', ...
       'Position', [100, 100, 1100, 800], 'Color', 'w');

% --- Subplot 1: 2D Trajectory & Obstacle Evasion ---
subplot(2, 2, 1);
plot(waypoints(:,1), waypoints(:,2), 'k--', 'LineWidth', 1.5, 'DisplayName', 'Nominal Waypoints');
hold on;
plot(log_ego_x, log_ego_y, 'b-', 'LineWidth', 2.0, 'DisplayName', 'Ego Vehicle Path');
plot(obs_true_x, obs_true_y, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'DisplayName', 'Obstacle');
xlabel('X Position (m)'); ylabel('Y Position (m)');
title('\bf 1. Vehicle Trajectory & Evasion Path');
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

% --- Subplot 3: Raw Sensor Measurements vs Kalman Filtered Distance ---
subplot(2, 2, 3);
valid_idx = log_raw_meas_x > 0;
plot(t(valid_idx), log_raw_meas_x(valid_idx), 'r.', 'MarkerSize', 8, 'DisplayName', 'Noisy Sensor Measurement');
hold on;
plot(t, log_kf_x, 'g-', 'LineWidth', 2.0, 'DisplayName', 'Kalman Filtered Distance');
xlabel('Time (s)'); ylabel('Longitudinal Distance (m)');
title('\bf 3. Kalman Filter Noise Reduction');
legend('Location', 'best'); grid on;

% --- Subplot 4: Dynamic Lateral Offset & Steering Control ---
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

disp('All plots generated successfully.');
