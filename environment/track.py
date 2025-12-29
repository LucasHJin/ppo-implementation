import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

class Track:
    TRACK_WIDTH = 6.0
    
    def __init__(self):
        self.control_points = np.array([
            [0, 0],
            [50, 0],
            [70, 20],
            [60, 40],
            [70, 50],
            [50, 70],
            [20, 70],
            [10, 50],
            [10, 20],
            [0, 10],
        ])
        self.waypoints = self.generate_waypoints() # points to approximate location on track
        self.normals = self.compute_normals()
        self.left_boundary = self.waypoints + self.normals * Track.TRACK_WIDTH
        self.right_boundary = self.waypoints - self.normals * Track.TRACK_WIDTH
        
    def generate_waypoints(self, factor=30):
        # close points loop
        points = np.vstack((self.control_points, self.control_points[0]))
        
        # parametric value for creating curves for points (euclidean distance) -> add on 0 for starting point
        t = np.concatenate(([0], np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))))
        cs_x = CubicSpline(t, points[:, 0], bc_type='periodic')
        cs_y = CubicSpline(t, points[:, 1], bc_type='periodic')
        
        # generate waypoints
        num_waypoints = len(self.control_points) * factor
        t_waypoints = np.linspace(0, t[-1], num_waypoints, endpoint=False)
        t_x = cs_x(t_waypoints)
        t_y = cs_y(t_waypoints)
        waypoints = np.column_stack((t_x, t_y))
        return waypoints
    
    def compute_normals(self):
        tangents = np.diff(self.waypoints, axis=0, append=[self.waypoints[0]]) # vectors pointing between waypoints
        tangent_lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangent_lengths = np.where(tangent_lengths==0, 1, tangent_lengths) # get rid of div by 0
        tangents = tangents / tangent_lengths
        
        normals = np.column_stack((-tangents[:, 1], tangents[:, 0])) # reverse to get normals
        return normals
    
    def visualize(self):
        fig, ax = plt.subplots(figsize=(10, 10)) # ax is plotting area, fig can have multiple axes
        
        waypoints_closed = np.vstack([self.waypoints, self.waypoints[0]])
        left_closed = np.vstack([self.left_boundary, self.left_boundary[0]])
        right_closed = np.vstack([self.right_boundary, self.right_boundary[0]])
        
        ax.plot(waypoints_closed[:, 0], waypoints_closed[:, 1], 'b-', linewidth=2)
        ax.plot(left_closed[:, 0], left_closed[:, 1], 'k-')
        ax.plot(right_closed[:, 0], right_closed[:, 1], 'k-')
        ax.set_aspect('equal') # equal axis scales
        
        plt.show()
        
track = Track()
track.visualize()