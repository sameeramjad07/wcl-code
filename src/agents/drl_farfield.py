"""
DRL Agent for Far-Field Only Regime
====================================

Specialized SAC/PPO agent trained exclusively for far-field beamsteering.
Always uses far-field phase profiles and full aperture.

Author: Research Team
Date: January 2026
"""

import numpy as np
from typing import Optional, Dict, Tuple
import torch

from src.agents.sac_agent import SACAgent
from src.agents.ppo_agent import PPOAgent
from src.core.system_model import SystemSimulator
from src.environment.ris_env import RISEnvironment


class DRLFarField:
    """DRL agent specialized for far-field optimization."""
    
    def __init__(
        self,
        simulator: SystemSimulator,
        agent_type: str = 'sac',  # 'sac' or 'ppo'
        aperture_size: int = 4096,  # Default full aperture
        hidden_dim: int = 256,
        lr: float = 3e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize far-field DRL agent.
        
        Args:
            simulator: System simulator
            agent_type: 'sac' or 'ppo'
            aperture_size: Number of active RIS elements (usually full 4096)
            hidden_dim: Hidden layer dimension
            lr: Learning rate
            device: Computation device
        """
        self.simulator = simulator
        self.agent_type = agent_type
        self.aperture_size = aperture_size
        self.device = device
        
        # Generate aperture mask (usually None for full aperture)
        self.aperture_mask = None if aperture_size >= simulator.config.N_total else \
                            simulator.ris_controller.generate_aperture_mask(aperture_size)
        
        # Create environment with far-field reward weights
        self.env = RISEnvironment(
            config=simulator.config,
            snapshot_type='A',
            aperture_mask=self.aperture_mask,
            reward_weights=(1.0, 1.0, 0.1),  # Lower interference penalty for far-field
            normalize_state=True
        )
        
        # Initialize DRL agent
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        if agent_type == 'sac':
            self.agent = SACAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                lr=lr,
                device=device
            )
        elif agent_type == 'ppo':
            self.agent = PPOAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                lr=lr,
                device=device
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def train(
        self,
        n_episodes: int = 1000,
        max_steps: int = 100,
        batch_size: int = 64,
        distance_range: Tuple[float, float] = (20.0, 50.0),
        verbose: bool = True
    ) -> Dict:
        """
        Train far-field DRL agent.
        
        Args:
            n_episodes: Number of training episodes
            max_steps: Max steps per episode
            batch_size: Batch size for updates
            distance_range: Range of user distances to train on (far-field)
            verbose: Print progress
            
        Returns:
            training_history: Dictionary with training metrics
        """
        episode_rewards = []
        episode_sum_rates = []
        
        for episode in range(n_episodes):
            # Random distance in far-field range
            r_A = np.random.uniform(*distance_range)
            
            state, _ = self.env.reset(options={'r_A': r_A})
            episode_reward = 0
            episode_sum_rate = 0
            
            for step in range(max_steps):
                # Select action
                if self.agent_type == 'sac':
                    action = self.agent.select_action(state, deterministic=False)
                else:
                    action, log_prob, value = self.agent.select_action(state, deterministic=False)
                
                # Environment step
                next_state, reward, done, truncated, info = self.env.step(action)
                
                # Store transition
                if self.agent_type == 'sac':
                    self.agent.replay_buffer.push(state, action, reward, next_state, done)
                    # Update SAC
                    if len(self.agent.replay_buffer) > batch_size:
                        self.agent.update(batch_size)
                else:
                    # Store PPO transition
                    self.agent.store_transition(state, action, reward, value, log_prob, done)
                
                episode_reward += reward
                episode_sum_rate = info['sum_rate']  # Track latest
                state = next_state
                
                if done or truncated:
                    break
            
            # Update PPO at end of episode
            if self.agent_type == 'ppo':
                self.agent.update(next_state)
            
            episode_rewards.append(episode_reward)
            episode_sum_rates.append(episode_sum_rate)
            
            # Logging
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_sum_rate = np.mean(episode_sum_rates[-100:])
                print(f"[Far-Field DRL] Episode {episode+1}/{n_episodes} | "
                      f"Avg Reward: {avg_reward:.3f} | Avg Sum-Rate: {avg_sum_rate:.3f} bps/Hz")
        
        return {
            'episode_rewards': episode_rewards,
            'episode_sum_rates': episode_sum_rates,
            'final_avg_reward': np.mean(episode_rewards[-100:]),
            'final_avg_sum_rate': np.mean(episode_sum_rates[-100:])
        }
    
    def optimize(
        self,
        user_A_distance: float,
        deterministic: bool = True
    ) -> Dict:
        """
        Optimize RIS configuration for given user distance.
        
        Args:
            user_A_distance: Distance of user A
            deterministic: Use deterministic policy
            
        Returns:
            result: Dictionary with phase_shifts, sum_rate, rates
        """
        # Set snapshot
        self.simulator.set_snapshot_A(user_A_distance)
        
        # Reset environment
        state, _ = self.env.reset(options={'r_A': user_A_distance})
        
        # Get action from agent
        if self.agent_type == 'sac':
            action = self.agent.select_action(state, deterministic=deterministic)
        else:
            action, _, _ = self.agent.select_action(state, deterministic=deterministic)
        
        # Execute action
        _, _, _, _, info = self.env.step(action)
        
        return {
            'sum_rate': info['sum_rate'],
            'rate_A': info['rate_A'],
            'rate_B': info['rate_B'],
            'interference': info['interference'],
            'aperture_size': self.aperture_size
        }
    
    def save(self, filepath: str):
        """Save agent."""
        self.agent.save(filepath)
    
    def load(self, filepath: str):
        """Load agent."""
        self.agent.load(filepath)