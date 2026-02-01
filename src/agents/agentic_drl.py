"""
Adaptive DRL with Threshold-Based Switching
============================================

Switches between near-field and far-field DRL agents based on
distance to far-field boundary (Rayleigh distance).

Author: Research Team
Date: January 2026
"""

import numpy as np
from typing import Dict, Optional
import torch

from src.agents.drl_nearfield import DRLNearField
from src.agents.drl_farfield import DRLFarField
from src.core.system_model import SystemSimulator


class AdaptiveDRL:
    """Adaptive DRL agent with threshold-based regime switching."""
    
    def __init__(
        self,
        simulator: SystemSimulator,
        agent_type: str = 'sac',
        near_field_aperture: int = 1024,
        far_field_aperture: int = 4096,
        threshold_factor: float = 1.0,  # Multiplier for d_FF
        hidden_dim: int = 256,
        lr: float = 3e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize adaptive DRL agent.
        
        Args:
            simulator: System simulator
            agent_type: 'sac' or 'ppo'
            near_field_aperture: Aperture size for near-field agent
            far_field_aperture: Aperture size for far-field agent
            threshold_factor: Multiplier for d_FF threshold
            hidden_dim: Hidden layer dimension
            lr: Learning rate
            device: Computation device
        """
        self.simulator = simulator
        self.agent_type = agent_type
        self.threshold_distance = threshold_factor * simulator.config.d_FF
        
        # Initialize both agents
        self.near_field_agent = DRLNearField(
            simulator=simulator,
            agent_type=agent_type,
            aperture_size=near_field_aperture,
            hidden_dim=hidden_dim,
            lr=lr,
            device=device
        )
        
        self.far_field_agent = DRLFarField(
            simulator=simulator,
            agent_type=agent_type,
            aperture_size=far_field_aperture,
            hidden_dim=hidden_dim,
            lr=lr,
            device=device
        )
        
        # Statistics
        self.regime_counts = {'near_field': 0, 'far_field': 0}
    
    def _select_regime(self, user_A_distance: float) -> str:
        """
        Determine regime based on distance threshold.
        
        Args:
            user_A_distance: Distance of user A
            
        Returns:
            regime: 'near_field' or 'far_field'
        """
        if user_A_distance < self.threshold_distance:
            return 'near_field'
        else:
            return 'far_field'
    
    def train(
        self,
        n_episodes_per_regime: int = 1000,
        max_steps: int = 100,
        batch_size: int = 64,
        verbose: bool = True
    ) -> Dict:
        """
        Train both near-field and far-field agents.
        
        Args:
            n_episodes_per_regime: Episodes to train each agent
            max_steps: Max steps per episode
            batch_size: Batch size for updates
            verbose: Print progress
            
        Returns:
            training_history: Combined training metrics
        """
        if verbose:
            print("="*70)
            print("Training Adaptive DRL Agent")
            print(f"Threshold distance: {self.threshold_distance:.2f}m (d_FF)")
            print("="*70)
        
        # Train near-field agent
        if verbose:
            print("\n[1/2] Training Near-Field Agent...")
        nf_history = self.near_field_agent.train(
            n_episodes=n_episodes_per_regime,
            max_steps=max_steps,
            batch_size=batch_size,
            distance_range=(2.0, self.threshold_distance - 1.0),
            verbose=verbose
        )
        
        # Train far-field agent
        if verbose:
            print("\n[2/2] Training Far-Field Agent...")
        ff_history = self.far_field_agent.train(
            n_episodes=n_episodes_per_regime,
            max_steps=max_steps,
            batch_size=batch_size,
            distance_range=(self.threshold_distance, 50.0),
            verbose=verbose
        )
        
        if verbose:
            print("\n" + "="*70)
            print("Adaptive DRL Training Complete!")
            print(f"Near-Field Final Avg Sum-Rate: {nf_history['final_avg_sum_rate']:.3f} bps/Hz")
            print(f"Far-Field Final Avg Sum-Rate: {ff_history['final_avg_sum_rate']:.3f} bps/Hz")
            print("="*70)
        
        return {
            'near_field': nf_history,
            'far_field': ff_history
        }
    
    def optimize(
        self,
        user_A_distance: float,
        deterministic: bool = True
    ) -> Dict:
        """
        Optimize RIS configuration using appropriate agent.
        
        Args:
            user_A_distance: Distance of user A
            deterministic: Use deterministic policy
            
        Returns:
            result: Dictionary with optimization results + regime info
        """
        # Select regime
        regime = self._select_regime(user_A_distance)
        self.regime_counts[regime] += 1
        
        # Use appropriate agent
        if regime == 'near_field':
            result = self.near_field_agent.optimize(user_A_distance, deterministic)
        else:
            result = self.far_field_agent.optimize(user_A_distance, deterministic)
        
        # Add regime info
        result['regime'] = regime
        result['threshold_distance'] = self.threshold_distance
        
        return result
    
    def evaluate(
        self,
        test_distances: np.ndarray,
        n_runs: int = 5
    ) -> Dict:
        """
        Evaluate adaptive agent across distance range.
        
        Args:
            test_distances: Array of distances to test
            n_runs: Number of runs per distance
            
        Returns:
            results: Dictionary with evaluation metrics
        """
        sum_rates = []
        regimes = []
        fairness = []
        
        for distance in test_distances:
            run_sum_rates = []
            run_regime = None
            
            for run in range(n_runs):
                result = self.optimize(distance, deterministic=True)
                run_sum_rates.append(result['sum_rate'])
                run_regime = result['regime']
            
            sum_rates.append(np.mean(run_sum_rates))
            regimes.append(run_regime)
            
            # Compute fairness from last run
            rate_A = result['rate_A']
            rate_B = result['rate_B']
            fairness_idx = (rate_A + rate_B)**2 / (2 * (rate_A**2 + rate_B**2))
            fairness.append(fairness_idx)
        
        return {
            'distances': test_distances,
            'sum_rates': np.array(sum_rates),
            'regimes': regimes,
            'fairness': np.array(fairness),
            'regime_counts': self.regime_counts.copy()
        }
    
    def save(self, filepath_prefix: str):
        """
        Save both agents.
        
        Args:
            filepath_prefix: Prefix for save paths
        """
        self.near_field_agent.save(f"{filepath_prefix}_nearfield.pt")
        self.far_field_agent.save(f"{filepath_prefix}_farfield.pt")
    
    def load(self, filepath_prefix: str):
        """
        Load both agents.
        
        Args:
            filepath_prefix: Prefix for load paths
        """
        self.near_field_agent.load(f"{filepath_prefix}_nearfield.pt")
        self.far_field_agent.load(f"{filepath_prefix}_farfield.pt")
    
    def get_statistics(self) -> Dict:
        """Get regime usage statistics."""
        total = sum(self.regime_counts.values())
        if total == 0:
            return {'near_field_pct': 0, 'far_field_pct': 0, 'total_calls': 0}
        
        return {
            'near_field_pct': 100 * self.regime_counts['near_field'] / total,
            'far_field_pct': 100 * self.regime_counts['far_field'] / total,
            'total_calls': total,
            'threshold_distance': self.threshold_distance
        }