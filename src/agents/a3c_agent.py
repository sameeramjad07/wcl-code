"""
Asynchronous Advantage Actor-Critic (A3C) Agent
================================================

Parallel training with multiple workers for RIS optimization.
Note: Simplified single-worker version for compatibility.

Author: Research Team
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import Tuple
import torch.multiprocessing as mp


class ActorCriticNetwork(nn.Module):
    """Shared actor-critic network for A3C."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Actor head
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Policy
        mean = torch.tanh(self.actor_mean(x))
        std = self.actor_log_std.exp().expand_as(mean)
        
        # Value
        value = self.critic(x)
        
        return mean, std, value
    
    def act(self, state, deterministic=False):
        mean, std, value = self.forward(state)
        
        if deterministic:
            action = mean
            log_prob = None
        else:
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, value


class A3CAgent:
    """A3C agent (simplified single-worker version)."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr: float = 1e-4,
        gamma: float = 0.99,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 40.0,
        n_steps: int = 20,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize A3C agent.
        
        Args:
            n_steps: Number of steps for n-step returns
        """
        self.device = device
        self.gamma = gamma
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        
        # Shared network
        self.network = ActorCriticNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.network.share_memory()  # Enable multiprocessing (if needed)
        
        # Optimizer
        self.optimizer = optim.RMSprop(self.network.parameters(), lr=lr, alpha=0.99, eps=1e-5)
        
        # Rollout storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def select_action(self, state: np.ndarray, deterministic: bool = False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.network.act(state, deterministic)
        
        return (
            action.cpu().numpy()[0],
            log_prob.cpu().item() if log_prob is not None else None,
            value.cpu().item()
        )
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_returns(self, next_value: float) -> torch.Tensor:
        """Compute n-step returns."""
        returns = []
        R = next_value
        
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            R = reward + self.gamma * R * (1 - done)
            returns.insert(0, R)
        
        return torch.FloatTensor(returns).to(self.device)
    
    def update(self, next_state: np.ndarray) -> dict:
        """Update network using accumulated rollout."""
        if len(self.states) == 0:
            return {}
        
        # Compute next value
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, next_value = self.network.act(next_state_tensor)
            next_value = next_value.cpu().item()
        
        # Compute returns
        returns = self.compute_returns(next_value)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        old_values = torch.FloatTensor(self.values).to(self.device)
        
        # Forward pass
        mean, std, values = self.network(states)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        
        # Advantages
        advantages = returns - old_values
        
        # Losses
        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(values.squeeze(), returns)
        
        # Total loss
        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Clear storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }
    
    def save(self, filepath: str):
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])