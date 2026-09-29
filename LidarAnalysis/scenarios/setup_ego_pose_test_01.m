%% Temporary ego vehicle pose bus

clear EgoPoseBus EgoPoseBusElements

% ActorID
e(1) = Simulink.BusElement;
e(1).Name = 'ActorID';
e(1).DataType = 'double';
e(1).Dimensions = 1;

% Position [x y z]
e(2) = Simulink.BusElement;
e(2).Name = 'Position';
e(2).DataType = 'double';
e(2).Dimensions = 3;

% Velocity [vx vy vz]
e(3) = Simulink.BusElement;
e(3).Name = 'Velocity';
e(3).DataType = 'double';
e(3).Dimensions = 3;

% Roll
e(4) = Simulink.BusElement;
e(4).Name = 'Roll';
e(4).DataType = 'double';
e(4).Dimensions = 1;

% Pitch
e(5) = Simulink.BusElement;
e(5).Name = 'Pitch';
e(5).DataType = 'double';
e(5).Dimensions = 1;

% Yaw
e(6) = Simulink.BusElement;
e(6).Name = 'Yaw';
e(6).DataType = 'double';
e(6).Dimensions = 1;

% Angular velocity [wx wy wz]
e(7) = Simulink.BusElement;
e(7).Name = 'AngularVelocity';
e(7).DataType = 'double';
e(7).Dimensions = 3;

% Create bus
EgoPoseBus = Simulink.Bus;
EgoPoseBus.Elements = e;