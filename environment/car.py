import numpy as np

class Car:
    MAX_SPEED = 30.0
    ACCELERATION = 10.0
    STEERING_SPEED = 3.0
    DRAG = 0.985 # friction in forward direction
    LATERAL_FRICTION = 0.85 # friction in sideways direction (drift)
    GRIP = 0.9 # how much power can be exerted laterally (also drift)
    LENGTH = 4.0
    WIDTH = 2.0
    
    def __init__(self, track):
        self.track = track
        self.reset()
    
    def reset(self):
        self.x, self.y, self.angle = self.track.get_start_pos()
        self.vx = 0.0
        self.vy = 0.0
        self.angular_velocity = 0.0
        self.progress = 0.0
        self.crashed = False
        self.finished = False
    
    def get_corners(self):
        half_length = Car.LENGTH / 2
        half_width = Car.WIDTH / 2
        
        # local frame
        local_corners = np.array([
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width]
        ])
        
        # convert to global frame with rotation matrix
        cos_a = np.cos(self.angle)
        sin_a = np.sin(self.angle)
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        world_corners = (rotation @ local_corners.T).T + np.array([self.x, self.y])
        return world_corners
    
    def update(self, steering, throttle, dt=0.05):
        """
        steering -> [-1.0, 1.0] for full left/full right
        throttle -> [0.0, 1.0] for power amount
        dt -> timestep
        """
        if self.crashed:
            return
        
        angular_velocity = steering * Car.STEERING_SPEED
        self.angle = self.angle + (angular_velocity * dt)
        self.angle = self.angle % (2 * np.pi) # keep in [0, 2π]
        
        # compute velocities relative to the car
        v_forward = self.vx * np.cos(self.angle) + self.vy * np.sin(self.angle)
        v_lateral = self.vx * (-np.sin(self.angle)) + self.vy * np.cos(self.angle)
        accel_forward = throttle * Car.ACCELERATION
        v_forward = (v_forward + (accel_forward * dt)) * Car.DRAG
        v_lateral = v_lateral * Car.LATERAL_FRICTION * Car.GRIP
        
        # convert back to global
        self.vx = v_forward * np.cos(self.angle) - v_lateral * np.sin(self.angle)
        self.vy = v_forward * np.sin(self.angle) + v_lateral * np.cos(self.angle)
        
        # clamp speed
        speed = np.sqrt((self.vx ** 2) + (self.vy ** 2))
        if speed > Car.MAX_SPEED:
            scale = Car.MAX_SPEED / speed
            self.vx *= scale
            self.vy *= scale
            
        # final updates
        self.x = self.x + (self.vx * dt)
        self.y = self.y + (self.vy * dt)
        self.progress = self.track.calc_progress(self.x, self.y)
        self.crashed = self.track.check_collision(self.get_corners())