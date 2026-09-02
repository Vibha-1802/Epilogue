import math

class VehicleControllerNode:
    def __init__(self, v_initial=10.0, v_target=10.0, v_min=2.0, wheelbase=2.5):
        self.v_current = v_initial
        self.v_target = v_target
        self.v_min = v_min
        self.wheelbase = wheelbase
        
    def calculate_controls(self, y_target, is_critical, dt, time_horizon):
        """Returns (accel, steer_angle) and updates internal state"""
        peak_steer = (y_target * 2.0 * math.pi * self.wheelbase) / (max(self.v_current, 0.1) * (time_horizon ** 2))
        steer_angle = peak_steer * math.sin(2 * math.pi * (dt / time_horizon))
        
        if is_critical:
            accel = -3.0
        else:
            if self.v_current < self.v_target:
                accel = 1.0
            else:
                accel = 0.0
                
        v_next = self.v_current + accel * dt
        
        if v_next <= self.v_min:
            v_next = self.v_min
            accel = (v_next - self.v_current) / dt 
        elif v_next > self.v_target:
            v_next = self.v_target
            accel = (v_next - self.v_current) / dt
            
        self.v_current = v_next
        
        return accel, steer_angle
