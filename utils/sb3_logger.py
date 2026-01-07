from stable_baselines3.common.callbacks import BaseCallback
import json

class TrainingLoggerCallback(BaseCallback):
    def __init__(self, save_path="data/training_info_sb3.json", verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.training_info = {'steps': [], 'rewards': []}
    
    def _on_step(self) -> bool:
        return True # just return true -> it's a required function for basecall
    
    def _on_rollout_end(self) -> None:
        if len(self.model.ep_info_buffer) > 0: # type: ignore
            # mean reward for rollout (once per rollout)
            mean_reward = sum(ep['r'] for ep in self.model.ep_info_buffer) / len(self.model.ep_info_buffer) # type: ignore
            self.training_info['steps'].append(self.num_timesteps)
            self.training_info['rewards'].append(float(mean_reward))
    
    def _on_training_end(self) -> None:
        try:
            with open(self.save_path, 'w') as f:
                json.dump(self.training_info, f, indent=2) # save
            print(f"\nTraining data saved to {self.save_path}")
        except Exception as e:
            print(f"Warning: Could not save training data: {e}")