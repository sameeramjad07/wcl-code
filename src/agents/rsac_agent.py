"""
Recurrent Soft Actor-Critic (RSAC) Agent
=========================================

SAC with LSTM for temporal dependencies in RIS optimization.
Useful for tracking moving users or time-varying channels.

Author: Research Team
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import Tuple, Optional
from collections import deque
import random


class RecurrentGaussianPolicy(nn.Module):
    """Recurrent policy with LSTM."""
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dim: int = 256,
        lstm_hidden_dim: int = 128,
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.lstm_hidden_dim = lstm_hidden_dim
        
        # LSTM layer for temporal processing
        self.lstm = nn.LSTM(state_dim, lstm_hidden_dim, batch_first=True)
        
        # Policy network
        self.fc1 = nn.Linear(lstm_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # Hidden state
        self.hidden_state = None
        
    def forward(self, state: torch.Tensor, hidden: Optional[Tuple] = None):
        """Forward pass with LSTM."""
        # state shape: (batch, seq_len, state_dim) or (batch, state_dim)
        if state.dim() == 2:
            state = state.unsqueeze(1)  # Add sequence dimension
        
        # LSTM forward
        lstm_out, hidden_new = self.lstm(state, hidden)
        
        # Take last output
        x = lstm_out[:, -1, :]
        
        # Policy network
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std, hidden_new
    
    def sample(self, state: torch.Tensor, hidden: Optional[Tuple] = None, deterministic: bool = False):
        """Sample action with temporal context."""
        mean, log_std, hidden_new = self.forward(state, hidden)
        std = log_std.exp()
        
        if deterministic:
            action = torch.tanh(mean)
            log_prob = None
        else:
            normal = Normal(mean, std)
            z = normal.rsample()
            action = torch.tanh(z)
            log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob, hidden_new
    
    def reset_hidden(self, batch_size: int = 1):
        """Reset LSTM hidden state."""
        device = next(self.parameters()).device
        self.hidden_state = (
            torch.zeros(1, batch_size, self.lstm_hidden_dim).to(device),
            torch.zeros(1, batch_size, self.lstm_hidden_dim).to(device)
        )
        return self.hidden_state


class RecurrentQNetwork(nn.Module):
    """Recurrent Q-network with LSTM."""
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dim: int = 256,
        lstm_hidden_dim: int = 128
    ):
        super().__init__()
        
        self.lstm_hidden_dim = lstm_hidden_dim
        
        # LSTM for state sequence
        self.lstm = nn.LSTM(state_dim + action_dim, lstm_hidden_dim, batch_first=True)
        
        # Q-value network
        self.fc1 = nn.Linear(lstm_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor, hidden: Optional[Tuple] = None):
        """Forward pass."""
        # Concatenate state and action
        if state.dim() == 2:
            state = state.unsqueeze(1)
        if action.dim() == 2:
            action = action.unsqueeze(1)
        
        x = torch.cat([state, action], dim=-1)
        
        # LSTM forward
        lstm_out, hidden_new = self.lstm(x, hidden)
        x = lstm_out[:, -1, :]
        
        # Q-network
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        
        return q, hidden_new


class RSACAgent:
    """Recurrent Soft Actor-Critic agent."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lstm_hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        sequence_length: int = 8,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize RSAC agent.
        
        Args:
            sequence_length: Length of temporal sequences to use
        """
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.sequence_length = sequence_length
        
        # Policy
        self.policy = RecurrentGaussianPolicy(
            state_dim, action_dim, hidden_dim, lstm_hidden_dim
        ).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Twin Q-networks
        self.q1 = RecurrentQNetwork(state_dim, action_dim, hidden_dim, lstm_hidden_dim).to(device)
        self.q2 = RecurrentQNetwork(state_dim, action_dim, hidden_dim, lstm_hidden_dim).to(device)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)
        
        # Target networks
        self.q1_target = RecurrentQNetwork(state_dim, action_dim, hidden_dim, lstm_hidden_dim).to(device)
        self.q2_target = RecurrentQNetwork(state_dim, action_dim, hidden_dim, lstm_hidden_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Replay buffer stores sequences
        self.replay_buffer = deque(maxlen=50000)
        
        # Current episode buffer
        self.episode_buffer = []
        
        # Hidden states
        self.policy_hidden = None
        self.reset_hidden_states()
    
    def reset_hidden_states(self):
        """Reset all hidden states."""
        self.policy_hidden = self.policy.reset_hidden(1)
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action with temporal context."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, _, self.policy_hidden = self.policy.sample(
                state, self.policy_hidden, deterministic
            )
        
        return action.cpu().numpy()[0]
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in episode buffer."""
        self.episode_buffer.append((state, action, reward, next_state, done))
        
        # If episode done, store sequence in replay buffer
        if done:
            self._store_episode()
    
    def _store_episode(self):
        """Store episode as sequences in replay buffer."""
        episode_length = len(self.episode_buffer)
        
        # Create overlapping sequences
        for i in range(episode_length - self.sequence_length + 1):
            sequence = self.episode_buffer[i:i + self.sequence_length]
            self.replay_buffer.append(sequence)
        
        # Clear episode buffer
        self.episode_buffer = []
    
    def update(self, batch_size: int) -> dict:
        """Update networks using sequences."""
        if len(self.replay_buffer) < batch_size:
            return {}
        
        # Sample batch of sequences
        batch_sequences = random.sample(self.replay_buffer, batch_size)
        
        # Prepare batch data
        states_seq = []
        actions_seq = []
        rewards_seq = []
        next_states_seq = []
        dones_seq = []
        
        for sequence in batch_sequences:
            states, actions, rewards, next_states, dones = zip(*sequence)
            states_seq.append(np.array(states))
            actions_seq.append(np.array(actions))
            rewards_seq.append(rewards[-1])  # Use last reward
            next_states_seq.append(np.array(next_states))
            dones_seq.append(dones[-1])
        
        states_seq = torch.FloatTensor(np.array(states_seq)).to(self.device)
        actions_seq = torch.FloatTensor(np.array(actions_seq)).to(self.device)
        rewards = torch.FloatTensor(rewards_seq).unsqueeze(1).to(self.device)
        next_states_seq = torch.FloatTensor(np.array(next_states_seq)).to(self.device)
        dones = torch.FloatTensor(dones_seq).unsqueeze(1).to(self.device)
        
        # Update Q-functions
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.policy.sample(next_states_seq)
            q1_next, _ = self.q1_target(next_states_seq, next_actions)
            q2_next, _ = self.q2_target(next_states_seq, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + (1 - dones) * self.gamma * q_next
        
        q1_pred, _ = self.q1(states_seq, actions_seq)
        q2_pred, _ = self.q2(states_seq, actions_seq)
        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)
        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # Update policy
        new_actions, log_probs, _ = self.policy.sample(states_seq)
        q1_new, _ = self.q1(states_seq, new_actions)
        q2_new, _ = self.q2(states_seq, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.alpha * log_probs - q_new).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Soft update targets
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)
        
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'policy_loss': policy_loss.item()
        }
    
    def _soft_update(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
    
    def save(self, filepath: str):
        torch.save({
            'policy': self.policy.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.q1.load_state_dict(checkpoint['q1'])
        self.q2.load_state_dict(checkpoint['q2'])