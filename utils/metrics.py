import matplotlib.pyplot as plt
import numpy as np
import torch
import json

def normalize(vals):
    min_v = np.min(vals)
    max_v = np.max(vals)
    return (vals - min_v) / (max_v - min_v)

def eval_training(data, output_path):
    plt.figure(figsize=(12, 7))
    colors = ['blue', 'green', 'orange']
    for (name, d), color in zip(data.items(), colors):
        normalized = normalize(d["rewards"])
        plt.plot(d["steps"], normalized, label=name, linewidth=2, color=color, alpha=0.6)
    plt.xlabel("Training Steps")
    plt.ylabel("Normalized Rewards")
    plt.title("Learning Speed Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

def eval_single_agent(env, agent, device, max_steps=2000):
    obs, _ = env.reset()
    total_reward = 0
    step = 0
    info = {}
    
    # run episode
    for step in range(max_steps):
        # get action and step
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action, _, _, _ = agent.get_action_and_value(obs_tensor)
            action = action.cpu().numpy()[0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # if stop
        if terminated or truncated:
            break
    
    return {
        'total_reward': total_reward,
        'steps': step + 1,
        'progress': info['progress'],
        'finished': info['finished'],
        'crashed': info['crashed'],
        'speed': info['speed']
    }

def eval_multi_agent(env, agent, device, max_steps=3000):
    obs_dict, _ = env.reset()
    total_reward_0 = 0
    total_reward_1 = 0
    step = 0
    info_dict = {}
    
    # run episode
    for step in range(max_steps):
        # get action and step
        with torch.no_grad():
            obs_0_tensor = torch.FloatTensor(obs_dict["0"]).unsqueeze(0).to(device)
            action_0, _, _, _ = agent.get_action_and_value(obs_0_tensor)
            action_0 = action_0.cpu().numpy()[0]
            
            obs_1_tensor = torch.FloatTensor(obs_dict["1"]).unsqueeze(0).to(device)
            action_1, _, _, _ = agent.get_action_and_value(obs_1_tensor)
            action_1 = action_1.cpu().numpy()[0]
        
        actions = {
            "0": action_0,
            "1": action_1
        }
        
        obs_dict, reward_dict, done_dict, truncated, info_dict = env.step(actions)
        total_reward_0 += reward_dict["0"]
        total_reward_1 += reward_dict["1"]
        
        # if stop
        if done_dict["__all__"]:
            break
        
    if info_dict['0']['finished']:
        chosen_idx = '0'
        chosen_reward = total_reward_0
    elif info_dict['1']['finished']:
        chosen_idx = '1'
        chosen_reward = total_reward_1
    else:
        chosen_idx = '0'
        chosen_reward = total_reward_0
    
    
    return {
        'total_reward': chosen_reward,
        'progress': info_dict[chosen_idx]['progress'],
        'finished': info_dict[chosen_idx]['finished'],
        'crashed': info_dict[chosen_idx]['crashed'],
        'speed': info_dict[chosen_idx]['speed'],
        'placement': info_dict[chosen_idx].get('placement', None),
        'steps': step + 1
    }
    
def eval_sb3_agent(env, model, max_steps=2000):
    obs, _ = env.reset()
    total_reward = 0
    step = 0
    info = {}
    
    for step in range(max_steps):
        action, states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    return {
        'total_reward': total_reward,
        'steps': step + 1,
        'progress': info['progress'],
        'finished': info['finished'],
        'crashed': info['crashed'],
        'speed': info['speed']
    }
    
def display_comparison(results_files, labels, output_path):
    all_results = []
    for file in results_files:
        with open(file, 'r') as f:
            all_results.append(json.load(f))
    
    categories = ['Success Rate', 'Crash Rate', 'Avg Steps\n(normalized)', 'Avg Progress', 'Avg Speed\n(normalized)']
    max_steps = max(r['avg_steps'] for r in all_results)
    max_speed = max(r['avg_speed'] for r in all_results)
    
    data = []
    for result in all_results:
        data.append([
            result['success_rate'],
            result['crash_rate'],
            result['avg_steps'] / max_steps, 
            result['avg_progress'],
            result['avg_speed'] / max_speed 
        ])
    
    x = np.arange(len(categories))
    width = 0.8 / len(data) 
    
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ["blue", "cyan", "green", "red"]
    
    # plot bars for each agent
    for i, (agent_data, label) in enumerate(zip(data, labels)):
        offset = (i - len(data)/2 + 0.5) * width
        ax.bar(x + offset, agent_data, width, label=label, color=colors[i])
    
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Agent Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1) 
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

#display_comparison(["results/single_agent_results.json", "results/multi_agent_results.json"], ["Single", "Multi"], "data/comparison.png")