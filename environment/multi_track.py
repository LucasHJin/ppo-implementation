import numpy as np
from .track import Track

class MultiTrack(Track):
    def __init__(self, track_pool=None, track_id=None, track_width=None):
        if track_pool is not None:
            if track_id is None:
                track_id = np.random.randint(0, len(track_pool))
            control_points = track_pool[track_id]
            width = track_width[track_id] if track_width is not None else 5.0
        else:
            control_points = None
            width = None
        
        # initalize parent Track with selected parameters
        super().__init__(control_points, width)
    
    def raycast_with_cars(self, origin, direction, cars, max_dist=50.0):
        wall_dist = self.raycast(origin, direction, max_dist)
        ray_dir = np.array([np.cos(direction), np.sin(direction)])
        min_car_dist = max_dist
        
        for car in cars:
            # skip if same car (approximated)
            car_pos = np.array([car.x, car.y])
            if np.linalg.norm(car_pos - origin) < 0.5: 
                continue
            
            # check intersection with each edge of car
            corners = car.get_corners()
            for i in range(4):
                seg_start = corners[i]
                seg_end = corners[(i + 1) % 4]
                
                dist = self.ray_seg_intersection(origin, ray_dir, seg_start, seg_end)
                if dist is not None:
                    min_car_dist = min(min_car_dist, dist)
        
        return min(wall_dist, min_car_dist)
        
    def ray_seg_intersection(self, origin, ray_dir, seg_start, seg_end):
        # note -> don't need to vectorize (only 2 cars + 4 segments/car)
        v1 = origin - seg_start
        v2 = seg_end - seg_start
        v3 = np.array([-ray_dir[1], ray_dir[0]])
        
        dotp = np.dot(v2, v3)
        if abs(dotp) < 1e-10:
            return None
        
        t = np.cross(v2, v1) / dotp
        s = np.dot(v1, v3) / dotp
        
        if t >= 0 and 0 <= s <= 1:
            return t
        
        return None