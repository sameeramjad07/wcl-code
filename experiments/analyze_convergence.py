"""
Convergence Analysis for DRL Agents
====================================

Analyzes training convergence and determines optimal episode count.
"""

import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from src.core.system_model import SystemConfig, SystemSimulator
from src.environment.ris_env import RISEnvironment
from src.agents.sac_agent import SACAgent
from src.agents.ppo_agent import PPOAgent
from src.agents.ddpg_agent import DDPGAgent
from src.agents.td3_agent import TD3Agent
from src.agents.rsac_agent import RSACAgent
from src.agents.a3c_agent import A3CAgent


def train_and_track(agent_class, agent_name, env, n_episodes=3000):
    """Train agent and track convergence."""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = agent_class(state_dim, action_dim)
    
    episode_rewards = []
    episode_sum_rates = []
    moving_avg_window = 50
    
    print(f"\nTraining {agent_name}...")
    
    for episode in tqdm(range(n_episodes)):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(100):
            # Select action based on agent type
            if hasattr(agent, 'select_action'):
                if agent_name in ['SAC', 'DDPG', 'TD3']:
                    action = agent.select_action(state)
                else:  # PPO
                    action, log_prob, value = agent.select_action(state)
            
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store transition
            if hasattr(agent, 'replay_buffer'):
                agent.replay_buffer.append((state, action, reward, next_state, done))
            else:  # PPO
                agent.store_transition(state, action, reward, value, log_prob, done)
            
            # Update
            if hasattr(agent, 'update'):
                if len(getattr(agent, 'replay_buffer', [])) > 64:
                    agent.update(64)
            
            episode_reward += reward
            state = next_state
            
            if done or truncated:
                break
        
        # PPO update
        if agent_name == 'PPO':
            agent.update(next_state)
        
        episode_rewards.append(episode_reward)
        episode_sum_rates.append(info.get('sum_rate', 0))
        
        # Check convergence
        if episode >= moving_avg_window:
            recent_avg = np.mean(episode_rewards[-moving_avg_window:])
            if episode % 100 == 0:
                print(f"  Episode {episode}: Avg Reward = {recent_avg:.3f}")
    
    # Compute moving average
    moving_avg_rewards = np.convolve(
        episode_rewards, 
        np.ones(moving_avg_window)/moving_avg_window, 
        mode='valid'
    )
    
    return {
        'episodes': np.arange(len(episode_rewards)),
        'rewards': np.array(episode_rewards),
        'sum_rates': np.array(episode_sum_rates),
        'moving_avg': moving_avg_rewards,
        'moving_avg_episodes': np.arange(moving_avg_window-1, len(episode_rewards))
    }


def analyze_convergence():
    """Main convergence analysis."""
    
    # Setup
    config = SystemConfig()
    simulator = SystemSimulator(config)
    
    env = RISEnvironment(
        config=config,
        snapshot_type='A',
        aperture_mask=None,
        reward_weights=(1.0, 1.0, 0.3)
    )
    
    # Agents to test
    agents = {
        'SAC': SACAgent,
        'PPO': PPOAgent,
        'DDPG': DDPGAgent,
        'TD3': TD3Agent,
        'RSAC': RSACAgent,
        'A3C': A3CAgent,
    }
    
    results = {}
    
    # Train each agent
    for agent_name, agent_class in agents.items():
        results[agent_name] = train_and_track(agent_class, agent_name, env, n_episodes=2000)
    
    # Plot convergence
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Raw rewards
    ax = axes[0, 0]
    for agent_name, data in results.items():
        ax.plot(data['episodes'], data['rewards'], alpha=0.3, label=f'{agent_name} (raw)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Raw Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Moving average
    ax = axes[0, 1]
    for agent_name, data in results.items():
        ax.plot(data['moving_avg_episodes'], data['moving_avg'], 
                linewidth=2, label=agent_name)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Moving Avg Reward (window=50)')
    ax.set_title('Convergence Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Sum-rates
    ax = axes[1, 0]
    for agent_name, data in results.items():
        moving_avg_sr = np.convolve(
            data['sum_rates'], 
            np.ones(50)/50, 
            mode='valid'
        )
        ax.plot(np.arange(49, len(data['sum_rates'])), moving_avg_sr, 
                linewidth=2, label=agent_name)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Sum-Rate (bps/Hz)')
    ax.set_title('Sum-Rate Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Episodes to convergence
    ax = axes[1, 1]
    convergence_threshold = 0.95
    episodes_to_converge = []
    agent_names_converged = []
    
    for agent_name, data in results.items():
        ma = data['moving_avg']
        final_perf = np.mean(ma[-100:])
        target = convergence_threshold * final_perf
        
        converged_idx = np.where(ma >= target)[0]
        if len(converged_idx) > 0:
            episodes_to_converge.append(converged_idx[0] + 50)
            agent_names_converged.append(agent_name)
    
    ax.bar(agent_names_converged, episodes_to_converge)
    ax.set_ylabel('Episodes to 95% Convergence')
    ax.set_title('Convergence Speed Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    save_dir = Path('results/plots')
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / 'convergence_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved convergence plot to {save_dir / 'convergence_analysis.png'}")
    
    # Print statistics
    print("\n" + "="*70)
    print("CONVERGENCE ANALYSIS RESULTS")
    print("="*70)
    for agent_name, episodes in zip(agent_names_converged, episodes_to_converge):
        print(f"{agent_name:10s}: {episodes:4d} episodes to 95% convergence")
    print("="*70)
    
    return results


if __name__ == "__main__":
    analyze_convergence()