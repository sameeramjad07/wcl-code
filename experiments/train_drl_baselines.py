"""
Train DRL Baseline Agents
==========================

Train SAC and PPO agents for near-field and far-field scenarios.
"""

import sys
sys.path.append('.')

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import yaml
import argparse

from src.core.system_model import SystemConfig
from src.environment.ris_env import RISEnvironment
from src.agents.sac_agent import SACAgent
from src.agents.ppo_agent import PPOAgent
from src.utils.config import load_config, save_config


def train_sac_agent(
    env: RISEnvironment,
    n_episodes: int = 1000,
    batch_size: int = 64,
    save_path: str = None
):
    """Train SAC agent."""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = SACAgent(state_dim, action_dim)
    
    episode_rewards = []
    
    for episode in tqdm(range(n_episodes), desc="Training SAC"):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(100):  # Max steps per episode
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)
            
            episode_reward += reward
            state = next_state
            
            if done or truncated:
                break
        
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode+1}, Avg Reward: {avg_reward:.2f}")
    
    if save_path:
        agent.save(save_path)
        print(f"Saved agent to {save_path}")
    
    return agent, episode_rewards


def train_ppo_agent(
    env: RISEnvironment,
    n_episodes: int = 1000,
    steps_per_episode: int = 100,
    save_path: str = None
):
    """Train PPO agent."""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = PPOAgent(state_dim, action_dim)
    
    episode_rewards = []
    
    for episode in tqdm(range(n_episodes), desc="Training PPO"):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(steps_per_episode):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            
            agent.store_transition(state, action, reward, value, log_prob, done)
            
            episode_reward += reward
            state = next_state
            
            if done or truncated:
                break
        
        # Update policy
        agent.update(next_state)
        
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode+1}, Avg Reward: {avg_reward:.2f}")
    
    if save_path:
        agent.save(save_path)
        print(f"Saved agent to {save_path}")
    
    return agent, episode_rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/drl_config.yaml')
    parser.add_argument('--agent', type=str, default='sac', choices=['sac', 'ppo'])
    parser.add_argument('--regime', type=str, default='near', choices=['near', 'far'])
    args = parser.parse_args()
    
    # Load config
    config_dict = load_config(args.config)
    
    # Create system config
    sys_config = SystemConfig(**config_dict['system'])
    
    # Create environment
    aperture_size = 1024 if args.regime == 'near' else 4096
    aperture_mask = None  # Will use controller to generate
    
    env = RISEnvironment(
        config=sys_config,
        snapshot_type='A',
        aperture_mask=aperture_mask,
        reward_weights=(1.0, 1.0, 0.3)
    )
    
    # Train agent
    save_dir = Path(f"data/trained_models/{args.agent}_{args.regime}_field")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if args.agent == 'sac':
        agent, rewards = train_sac_agent(
            env,
            n_episodes=config_dict['training']['n_episodes'],
            save_path=str(save_dir / 'agent.pt')
        )
    else:
        agent, rewards = train_ppo_agent(
            env,
            n_episodes=config_dict['training']['n_episodes'],
            save_path=str(save_dir / 'agent.pt')
        )
    
    # Save training history
    np.save(save_dir / 'rewards.npy', rewards)
    print(f"Training complete! Saved to {save_dir}")


if __name__ == "__main__":
    main()