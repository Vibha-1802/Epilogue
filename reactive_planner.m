function [steering, throttle, brake, obstacle_status, closest_obstacle_x, risk_score, lateral_offset] = ...
    reactive_planner(fused_tracks, ego_x, ego_y, ego_yaw, goal_position, waypoints)
% =========================================================================
% ADVANCED REACTIVE PLANNER WITH THREE PILLARS:
%   1. KALMAN FILTERING   : Noisy track state estimation & relative velocity
%   2. RISK SCORING       : Spatio-temporal threat (TTC + Gaussian spatial decay)
%   3. LATERAL OFFSET     : Continuous dynamic path offset & smooth steering
% =========================================================================

% Default Outputs
steering           = 0;
throttle           = 0.4;
brake              = 0;
obstacle_status    = 0;
closest_obstacle_x = 999;
risk_score         = 0.0;
lateral_offset     = 0.0;

% Sampling time step (seconds)
dt = 0.1;

% =========================================================================
% PERSISTENT STATE FOR KALMAN FILTER & SMOOTH LATERAL OFFSET
% =========================================================================
persistent x_est P_est current_offset

% State vector x_est = [pos_x; pos_y; vel_x; vel_y] (Relative frame)
if isempty(x_est)
    x_est = [999; 0; 0; 0]; 
end

% Estimation Error Covariance Matrix P
if isempty(P_est)
    P_est = eye(4) * 1.0;
end

% Filtered Lateral Offset state for continuous steering transitions
if isempty(current_offset)
    current_offset = 0.0;
end

% =========================================================================
% PILLAR 1: KALMAN FILTERING (Track Estimation & Velocity Filtering)
% =========================================================================
% State Transition Matrix A for Constant Velocity (CV) Kinematic Model
A_kf = [1  0  dt  0;
        0  1  0  dt;
        0  0  1   0;
        0  0  0   1];

% Measurement Observation Matrix H (Measures relative px, py)
H_kf = [1 0 0 0;
        0 1 0 0]; 

% Process Noise Covariance Q & Measurement Noise Covariance R
Q_kf = eye(4) * 0.05; 
R_kf = eye(2) * 0.25;  

% --- Step 1A: Kalman Predict Step ---
x_pred = A_kf * x_est;
P_pred = A_kf * P_est * A_kf' + Q_kf;

% --- Extract Closest Track Measurement from fused_tracks ---
raw_meas_x = 999;
raw_meas_y = 0;
min_raw_dist = 999;

if size(fused_tracks, 2) >= 4 && ~isempty(fused_tracks)
    for i = 1:size(fused_tracks, 1)
        px = fused_tracks(i, 3);
        py = fused_tracks(i, 4);
        if px == 0 && py == 0
            continue; 
        end
        dist = sqrt(px^2 + py^2);
        if dist < min_raw_dist && px > -2.0 % Filter out objects behind ego
            min_raw_dist = dist;
            raw_meas_x = px;
            raw_meas_y = py;
        end
    end
end

% --- Step 1B: Kalman Update Step ---
if raw_meas_x < 150 % Valid sensor detection received
    z = [raw_meas_x; raw_meas_y];
    % Compute Kalman Gain K
    K_gain = P_pred * H_kf' / (H_kf * P_pred * H_kf' + R_kf);
    % State and Covariance Update
    x_est  = x_pred + K_gain * (z - H_kf * x_pred);
    P_est  = (eye(4) - K_gain * H_kf) * P_pred;
else
    % No measurement: Rely on Constant Velocity kinematic prediction
    x_est = x_pred;
    P_est = P_pred;
end

% Filtered relative states:
target_x  = x_est(1); % Relative longitudinal distance (m)
target_y  = x_est(2); % Relative lateral displacement (m)
rel_vel_x = x_est(3); % Relative longitudinal velocity (m/s)

closest_obstacle_x = target_x;

% =========================================================================
% PILLAR 2: SPATIO-TEMPORAL RISK SCORING MODEL
% =========================================================================
% Calculate Time-To-Collision (TTC)
closing_speed = -rel_vel_x; % Positive if target is closing in
if closing_speed > 0.1 && target_x > 0
    ttc = target_x / closing_speed;
else
    ttc = target_x / max(0.5, 4.0); % Fallback TTC estimation
end

% 2A. Spatial Risk (2D Anisotropic Gaussian Decay)
sigma_x = 10.0; % Longitudinal threat horizon (m)
sigma_y = 1.4;  % Lateral lane corridor boundary (m)
spatial_risk = exp(-0.5 * (target_x / sigma_x)^2) * exp(-0.5 * (target_y / sigma_y)^2);

% 2B. Temporal Risk (Exponential TTC Horizon Scaling)
ttc_tau = 3.0; % Critical time threshold (seconds)
if ttc < 10.0 && target_x > 0
    ttc_risk = exp(-ttc / ttc_tau);
else
    ttc_risk = 0.0;
end

% 2C. Total Weighted Risk Score [0.0, 1.0]
w_spatial  = 0.6;
w_ttc      = 0.4;
risk_score = min(1.0, max(0.0, w_spatial * spatial_risk + w_ttc * ttc_risk));

% Obstacle Hazard Flag Trigger
if risk_score > 0.35
    obstacle_status = 1;
end

% =========================================================================
% PILLAR 3: DYNAMIC LATERAL OFFSET TRAJECTORY PLANNING
% =========================================================================
LANE_CLEARANCE = 1.8; % Required lateral shift for clearance (m)

% Compute target lateral offset magnitude based on Risk Score and Obstacle Position
target_lateral_offset = 0.0;
if obstacle_status == 1 && target_x > 0 && target_x < 15.0
    % Select evasion side (evade left if obstacle is right, evade right if obstacle is center/left)
    if target_y >= 0
        evade_dir = -1.0; % Shift right
    else
        evade_dir = 1.0;  % Shift left
    end
    
    % Proportional lateral shift scaled by continuous Risk Score
    target_lateral_offset = evade_dir * LANE_CLEARANCE * risk_score;
end

% Low-pass filter for smooth lateral trajectory transition (prevents steering jerks)
alpha_offset   = 0.15;
current_offset = (1 - alpha_offset) * current_offset + alpha_offset * target_lateral_offset;
lateral_offset = current_offset;

% =========================================================================
% ROUTE WAYPOINT INTEGRATION & CLOSED-LOOP STEERING CONTROL
% =========================================================================
num_waypoints = size(waypoints, 1);
if num_waypoints < 2
    return;
end

% Find closest route waypoint
closest_wp  = 1;
min_wp_dist = 1e9;
for i = 1:num_waypoints
    dx_wp = waypoints(i,1) - ego_x;
    dy_wp = waypoints(i,2) - ego_y;
    d_wp  = sqrt(dx_wp*dx_wp + dy_wp*dy_wp);
    if d_wp < min_wp_dist
        min_wp_dist = d_wp;
        closest_wp  = i;
    end
end

% Target lookahead waypoint
target_wp = min(num_waypoints, closest_wp + 1);
wp_x = waypoints(target_wp, 1);
wp_y = waypoints(target_wp, 2);

% Nominal Route Direction Angle
dx_route  = wp_x - ego_x;
dy_route  = wp_y - ego_y;
route_yaw = atan2(dy_route, dx_route);

% Project Perpendicular Lateral Offset onto Waypoint Coordinates
offset_wp_x = wp_x - lateral_offset * sin(route_yaw);
offset_wp_y = wp_y + lateral_offset * cos(route_yaw);

% Heading Error calculation towards offset target
dx_offset = offset_wp_x - ego_x;
dy_offset = offset_wp_y - ego_y;
desired_heading = atan2(dy_offset, dx_offset);

heading_error = desired_heading - ego_yaw;

% Heading error normalization [-pi, pi]
while heading_error > pi,  heading_error = heading_error - 2*pi; end
while heading_error < -pi, heading_error = heading_error + 2*pi; end

% Proportional Steering Controller
STEERING_GAIN = 0.75;
MAX_STEERING  = 0.45;
steering = STEERING_GAIN * heading_error;
steering = max(-MAX_STEERING, min(MAX_STEERING, steering));

% =========================================================================
% SPEED COMMAND MODULATION (THROTTLE & BRAKE)
% =========================================================================
goal_x    = goal_position(1);
goal_y    = goal_position(2);
goal_dist = sqrt((goal_x - ego_x)^2 + (goal_y - ego_y)^2);

if goal_dist < 2.0
    steering = 0;
    throttle = 0;
    brake    = 0.8;
    return;
end

% Adaptive Longitudinal Speed Modulation based on Risk Score
if risk_score > 0.70
    % Severe Threat: Engage Active Braking
    throttle = 0.0;
    brake    = min(0.9, 0.4 + 0.5 * risk_score);
elseif risk_score > 0.35
    % Moderate Threat: Coast & Modulate Throttle during Evasion
    throttle = max(0.1, 0.4 * (1.0 - risk_score));
    brake    = 0.0;
else
    % Safe Corridor: Cruise Speed
    throttle = 0.4;
    brake    = 0.0;
end

end
