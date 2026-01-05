import random
import numpy as np
import torch
from environment.multi_racing_env import MultiRacingEnv
from environment.track import gen_tracks
from agent.self_play_ppo import SelfPlayPPO
from configs.self_play_config import hyperparams_config

def train():
    # set seeds for reproducibility
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    config = hyperparams_config()
    set_seed(config["seed"])
    
    print("Generating track pool")
    TRACK_POOL = gen_tracks(num_tracks=config["num_envs"], seed=config["seed"])
    TRACK_WIDTHS = [np.random.randint(6, 10) for _ in range(config["num_envs"])]
    TRACK_ASSIGNMENTS = [i for i in range(config["num_envs"])]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{'='*60}")
    print("SELF PLAY PPO TRAINING")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Total timesteps: {config['total_timesteps']:,}")
    print(f"Num environments: {config['num_envs']}")
    print(f"Batch size: {config['batch_size']:,}")
    print(f"Snapshot frequency: {config['snapshot_freq']}")
    print(f"Pool size: {config['pool_size']}")
    print(f"Expected updates: {config['total_timesteps'] // config['batch_size']}")
    print(f"{'='*60}\n")
    
    # factory function -> separate vectorized envs
    def env_fn(env_idx):
        track_id = TRACK_ASSIGNMENTS[env_idx]
        return MultiRacingEnv(num_agents=2, num_sensors=11, track_pool=TRACK_POOL, track_id=track_id, track_width=TRACK_WIDTHS)
    trainer = SelfPlayPPO(env_fn, config, device=device)
    
    print(f"\n{'='*60}")
    print("Starting training")
    print(f"{'='*60}\n")
    trainer.train()
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}\n")
    
    # save final model
    final_path = "models/self_play_agent_local.pth"
    trainer.save(final_path)
    print(f"Final model saved to {final_path}")
    
if __name__ == "__main__":
    train()