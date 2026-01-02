import torch
import numpy as np
import pygame
from environment.racing_env import RacingEnv
from agent.ppo import Agent

def convert_coords(x, y, ox, oy, scale, screen_size):
    screen_x = int((x + ox) * scale)
    screen_y = int(screen_size - (y + oy) * scale)
    return screen_x, screen_y

def eval(model_path="models/single_agent.pth", num_episodes=3):
    # setup
    env = RacingEnv()
    track = env.track
    agent = Agent(env.observation_space, env.action_space)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load trained weights
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.to(device)
    agent.eval()
    
    # pygame setup
    pygame.init()
    all_points = np.vstack([track.left_boundary, track.right_boundary]) # track bounds for scaling
    min_x, min_y = all_points.min(axis=0)
    max_x, max_y = all_points.max(axis=0)
    padding = 10
    track_width = max_x - min_x
    track_height = max_y - min_y
    screen_size = 800
    screen = pygame.display.set_mode((screen_size, screen_size))
    clock = pygame.time.Clock()
    scale = min(screen_size / (track_width + 2*padding), screen_size / (track_height + 2*padding))
    offset_x = -min_x + padding
    offset_y = -min_y + padding
    left_points = [convert_coords(p[0], p[1], offset_x, offset_y, scale, screen_size) for p in track.left_boundary]
    right_points = [convert_coords(p[0], p[1], offset_x, offset_y, scale, screen_size) for p in track.right_boundary]
    track_polygon = left_points + right_points[::-1] + [left_points[0]]
    start_left = convert_coords(
        track.waypoints[0][0] + track.normals[0][0] * track.track_width,
        track.waypoints[0][1] + track.normals[0][1] * track.track_width,
        offset_x,
        offset_y,
        scale,
        screen_size
    )
    start_right = convert_coords(
        track.waypoints[0][0] - track.normals[0][0] * track.track_width,
        track.waypoints[0][1] - track.normals[0][1] * track.track_width,
        offset_x,
        offset_y,
        scale,
        screen_size
    )
    
    print("Testing:")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        
        path_points = []
        total_reward = 0
        running = True
        
        # run episodes
        for step in range(2000):
            for event in pygame.event.get(): # handle quit
                if event.type == pygame.QUIT:
                    running = False
                    break
            
            if not running:
                break
            
            # get action and step
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action, _, _, _ = agent.get_action_and_value(obs_tensor)
                action = action.cpu().numpy()[0]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # update visualization
            screen.fill((50, 50, 50))
            pygame.draw.polygon(screen, (180, 180, 180), track_polygon)
            pygame.draw.lines(screen, (0, 0, 0), True, left_points, 4)
            pygame.draw.lines(screen, (0, 0, 0), True, right_points, 4)
            pygame.draw.line(screen, (0, 255, 0), start_left, start_right, 5)
            car = env.car
            path_points.append(convert_coords(car.x, car.y, offset_x, offset_y, scale, screen_size))
            if len(path_points) > 1:
                pygame.draw.lines(screen, (255, 100, 100), False, path_points, 3)  # Red trail
            corners = car.get_corners()
            car_screen_points = [convert_coords(c[0], c[1], offset_x, offset_y, scale, screen_size) for c in corners]
            pygame.draw.polygon(screen, (255, 0, 0), car_screen_points)  # Red car
            pygame.draw.polygon(screen, (150, 0, 0), car_screen_points, 2)  # Dark red outline
            font = pygame.font.Font(None, 30)
            info_text = [
                f"Episode: {episode + 1}/{num_episodes}",
                f"Step: {step}",
                f"Progress: {info['progress']:.1%}",
                f"Speed: {info['speed']:.1f}",
                f"Reward: {total_reward:.0f}"
            ]
            
            text_offset = 10
            for text in info_text:
                text_surface = font.render(text, True, (255, 255, 255))
                screen.blit(text_surface, (10, text_offset))
                text_offset += 25
                
            pygame.display.flip()
            clock.tick(60)
            
            # if stop
            if terminated or truncated:
                reason = "Finished" if info['finished'] else "Crashed" if info['crashed'] else "Time limit"
                print(f"Episode {episode+1}: {reason} | Steps: {step} | "
                      f"Reward: {total_reward:.1f} | Progress: {info['progress']:.1%}")
                pygame.time.wait(3000)
                break
        
        if not running:
            break
    
    pygame.quit()


if __name__ == "__main__":
    eval(num_episodes=3)