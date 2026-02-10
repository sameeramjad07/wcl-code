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

from src.core.system_model import SystemConfig, SystemSimulator
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
    parser.add_argument('--agent', type=str, default='sac', 
                       choices=['sac', 'ppo', 'ddpg', 'td3'])
    parser.add_argument('--regime', type=str, default='near', choices=['near', 'far'])
    parser.add_argument('--episodes', type=int, default=2000)
    args = parser.parse_args()
    
    # Load config
    try:
        config_dict = load_config(args.config)
    except:
        # Default config
        config_dict = {
            'system': {},
            'training': {'n_episodes': args.episodes, 'batch_size': 64}
        }
    
    # Create system config
    sys_config = SystemConfig()
    
    # Create environment
    aperture_size = 1024 if args.regime == 'near' else 4096
    reward_weights = (1.0, 1.0, 0.5) if args.regime == 'near' else (1.0, 1.0, 0.1)
    
    # Generate mask
    simulator = SystemSimulator(sys_config)
    mask = simulator.ris_controller.generate_aperture_mask(aperture_size)
    
    env = RISEnvironment(
        config=sys_config,
        snapshot_type='A',
        aperture_mask=mask,
        reward_weights=reward_weights
    )
    
    # Create save directory
    save_dir = Path(f"data/trained_models/{args.agent}_{args.regime}_field")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Training {args.agent.upper()} Agent - {args.regime.upper()} Field")
    print(f"{'='*70}")
    print(f"Aperture Size: {aperture_size}")
    print(f"Episodes: {args.episodes}")
    print(f"Save Directory: {save_dir}")
    print(f"{'='*70}\n")
    
    # Train agent
    if args.agent == 'sac':
        agent, rewards = train_sac_agent(
            env,
            n_episodes=args.episodes,
            save_path=str(save_dir / 'agent.pt')
        )
    elif args.agent == 'ppo':
        agent, rewards = train_ppo_agent(
            env,
            n_episodes=args.episodes,
            save_path=str(save_dir / 'agent.pt')
        )
    # Add other agents...
    
    # Save training history
    np.save(save_dir / 'rewards.npy', rewards)
    
    # Save metadata
    import json
    metadata = {
        'agent_type': args.agent,
        'regime': args.regime,
        'aperture_size': aperture_size,
        'n_episodes': args.episodes,
        'final_avg_reward': float(np.mean(rewards[-100:])),
        'reward_weights': reward_weights
    }
    with open(save_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✓ Training Complete!")
    print(f"  Final Avg Reward (last 100): {np.mean(rewards[-100:]):.3f}")
    print(f"  Saved to: {save_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()