import numpy as np
from .car import Car

class MultiCar(Car):
    def check_car_collision(self, cars):
        corners = self.get_corners()
        
        for other_car in cars:
            if other_car is self:
                continue
            other_corners = other_car.get_corners()
            if self.rectangles_intersect(corners, other_corners):
                return True
        return False
    
    def get_axes(self, corners):
        axes = []
        # note -> only need first 2 sides because rectangle is a parallelogram 
        for i in range(2):
            edge = corners[(i + 1) % 4] - corners[i]
            normal = np.array([-edge[1], edge[0]])
            axes.append(normal)
        return axes
        
    def rectangles_intersect(self, corners_a, corners_b):
        """
        https://programmerart.weebly.com/separating-axis-theorem.html
        1. Get perpindicular vectors for each edge
        2. Dot product each vertex and perpindicular vector for the projection
        3. If there is a gap (in any case) -> not colliding
        """
        axes_a = self.get_axes(corners_a)
        axes_b = self.get_axes(corners_b)
        axes = axes_a + axes_b
        
        for axis in axes:
            proj_a = [np.dot(corner, axis) for corner in corners_a]
            proj_b = [np.dot(corner, axis) for corner in corners_b]
            # check for gap
            if max(proj_a) < min(proj_b) or max(proj_b) < min(proj_a):
                return False
            
        return True