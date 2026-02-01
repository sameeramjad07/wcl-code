"""
Gymnasium Environment for RIS Optimization
===========================================

OpenAI Gym-style environment for training DRL agents.

Author: Research Team
Date: January 2026
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Optional

from src.core.system_model import SystemConfig, SystemSimulator
from src.core.channel_model import User


class RISEnvironment(gym.Env):
    """
    Custom Gym environment for XL-RIS optimization.
    
    State: [r_A, r_B, θ_A, θ_B, strategy_ID]
    Action: [phase_shifts (normalized), power_allocation_ratio]
    Reward: α*log(1+SINR_A) + β*log(1+SINR_B) - γ*I_leakage
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        config: SystemConfig,
        snapshot_type: str = 'A',  # 'A' or 'B'
        aperture_mask: Optional[np.ndarray] = None,
        reward_weights: Tuple[float, float, float] = (1.0, 1.0, 0.1),
        phase_quantization: Optional[int] = None,  # None for continuous
        normalize_state: bool = True
    ):
        """
        Initialize environment.
        
        Args:
            config: System configuration
            snapshot_type: 'A' (mixed-field) or 'B' (far-field NOMA)
            aperture_mask: Binary mask for active RIS elements
            reward_weights: (α, β, γ) for reward function
            phase_quantization: Number of phase levels (None = continuous)
            normalize_state: Whether to normalize state
        """
        super().__init__()
        
        self.config = config
        self.snapshot_type = snapshot_type
        self.aperture_mask = aperture_mask
        self.alpha, self.beta, self.gamma = reward_weights
        self.phase_quantization = phase_quantization
        self.normalize_state = normalize_state
        
        # Initialize simulator
        self.simulator = SystemSimulator(config)
        
        # Determine active elements
        if aperture_mask is not None:
            self.n_active = int(np.sum(aperture_mask))
            self.active_indices = np.where(aperture_mask == 1)[0]
        else:
            self.n_active = config.N_total
            self.active_indices = np.arange(config.N_total)
        
        # Define action and observation spaces
        # Action: [phase_shifts (n_active), power_ratio (1)]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_active + 1,),
            dtype=np.float32
        )
        
        # State: [r_A, r_B, θ_A, θ_B, strategy_ID]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),
            high=np.array([100, 100, 2*np.pi, 2*np.pi, 2]),
            dtype=np.float32
        )
        
        # Current state
        self.current_r_A = None
        self.state = None
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment."""
        super().reset(seed=seed)
        
        if options and 'r_A' in options:
            self.current_r_A = options['r_A']
        else:
            # Random position for Snapshot A
            if self.snapshot_type == 'A':
                self.current_r_A = np.random.uniform(2.0, 25.0)
            else:
                self.current_r_A = 45.0
        
        # Set snapshot
        if self.snapshot_type == 'A':
            self.simulator.set_snapshot_A(self.current_r_A)
        else:
            self.simulator.set_snapshot_B()
        
        # Determine strategy
        is_near_field = self.simulator.user_A.is_near_field(
            self.config.d_FF, self.config.RIS_position
        )
        strategy_id = 0 if is_near_field else 1
        
        # Create state
        self.state = np.array([
            self.simulator.user_A.r,
            self.simulator.user_B.r,
            self.simulator.user_A.theta,
            self.simulator.user_B.theta,
            strategy_id
        ], dtype=np.float32)
        
        if self.normalize_state:
            self.state = self._normalize_state(self.state)
        
        return self.state, {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute action and return results."""
        # Parse action
        phase_action = action[:-1]  # Phase shifts (normalized to [-1, 1])
        power_ratio = (action[-1] + 1) / 2  # Normalize to [0, 1]
        
        # Convert phase action to actual phase shifts
        phase_shifts_active = (phase_action + 1) * np.pi  # Map to [0, 2π]
        
        # Quantize if needed
        if self.phase_quantization is not None:
            levels = np.linspace(0, 2*np.pi, self.phase_quantization)
            phase_shifts_active = levels[np.digitize(phase_shifts_active, levels) - 1]
        
        # Create full phase shift array
        phase_shifts = np.zeros(self.config.N_total)
        phase_shifts[self.active_indices] = phase_shifts_active
        
        # Power allocation
        power_allocation = np.array([power_ratio, 1 - power_ratio])
        
        # Compute performance
        sum_rate, rate_A, rate_B = self.simulator.compute_sum_rate(
            phase_shifts, self.aperture_mask, power_allocation
        )
        
        # Compute interference leakage
        h_R_A = self.simulator.channel_model.compute_ris_to_user_channel(self.simulator.user_A)
        h_R_B = self.simulator.channel_model.compute_ris_to_user_channel(self.simulator.user_B)
        H_A = self.simulator.channel_model.compute_cascaded_channel(
            h_R_A, self.simulator.G, phase_shifts, self.aperture_mask
        )
        H_B = self.simulator.channel_model.compute_cascaded_channel(
            h_R_B, self.simulator.G, phase_shifts, self.aperture_mask
        )
        H_users = np.column_stack([H_A, H_B])
        W = self.simulator._compute_precoder(H_users, power_allocation)
        
        # Interference from A to B
        interference_A_to_B = np.abs(H_B.conj().T @ W[:, 0])**2
        interference_B_to_A = np.abs(H_A.conj().T @ W[:, 1])**2
        total_interference = interference_A_to_B + interference_B_to_A
        
        # Compute reward
        reward = (
            self.alpha * np.log(1 + 2**rate_A) +
            self.beta * np.log(1 + 2**rate_B) -
            self.gamma * total_interference
        )
        
        # Episode done (single-step for now)
        done = True
        truncated = False
        
        # Info
        info = {
            'sum_rate': sum_rate,
            'rate_A': rate_A,
            'rate_B': rate_B,
            'interference': total_interference,
            'power_ratio': power_ratio
        }
        
        return self.state, reward, done, truncated, info
    
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state to [0, 1]."""
        normalized = state.copy()
        normalized[0] /= 100.0  # r_A
        normalized[1] /= 100.0  # r_B
        normalized[2] /= (2 * np.pi)  # θ_A
        normalized[3] /= (2 * np.pi)  # θ_B
        normalized[4] /= 2.0  # strategy_ID
        return normalized
    
    def render(self):
        """Render environment (optional)."""
        pass