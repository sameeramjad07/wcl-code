"""
Evaluate All Methods and Generate Plots
========================================

Generate the 4 critical plots for the paper.
"""

import sys
sys.path.append('.')

import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm

from src.core.system_model import SystemConfig, SystemSimulator, BaselineOptimizer
from src.utils.visualization import (
    plot_sum_rate_vs_distance,
    plot_fairness_vs_distance,
    plot_aperture_influence
)
from src.utils.config import load_config


def evaluate_snapshot_a(config: SystemConfig, distances: np.ndarray):
    """Evaluate all methods on Snapshot A."""
    simulator = SystemSimulator(config)
    optimizer = BaselineOptimizer(simulator)
    
    results = {
        'Near-Field Only': {'sum_rates': [], 'fairness': []},
        'Far-Field Only': {'sum_rates': [], 'fairness': []},
        'Adaptive Threshold': {'sum_rates': [], 'fairness': []},
    }
    
    for r_A in tqdm(distances, desc="Evaluating distances"):
        simulator.set_snapshot_A(r_A)
        
        # Near-field only
        res_nf = optimizer.near_field_baseline()
        results['Near-Field Only']['sum_rates'].append(res_nf['sum_rate'])
        results['Near-Field Only']['fairness'].append(
            simulator.compute_jains_fairness(res_nf['rate_A'], res_nf['rate_B'])
        )
        
        # Far-field only
        res_ff = optimizer.far_field_baseline()
        results['Far-Field Only']['sum_rates'].append(res_ff['sum_rate'])
        results['Far-Field Only']['fairness'].append(
            simulator.compute_jains_fairness(res_ff['rate_A'], res_ff['rate_B'])
        )
        
        # Adaptive threshold
        res_adaptive = optimizer.adaptive_threshold_baseline()
        results['Adaptive Threshold']['sum_rates'].append(res_adaptive['sum_rate'])
        results['Adaptive Threshold']['fairness'].append(
            simulator.compute_jains_fairness(res_adaptive['rate_A'], res_adaptive['rate_B'])
        )
    
    return results


def evaluate_aperture_influence(config: SystemConfig):
    """Evaluate aperture size influence on SINR."""
    simulator = SystemSimulator(config)
    simulator.set_snapshot_A(r_A=2.0)  # User A very close
    
    aperture_sizes = [16*16, 24*24, 32*32, 48*48, 64*64]
    sinr_data = {'User A': [], 'User B': []}
    
    for size in tqdm(aperture_sizes, desc="Testing aperture sizes"):
        mask = simulator.ris_controller.generate_aperture_mask(size)
        phase_shifts = simulator.ris_controller.compute_near_field_phase_profile(
            simulator.user_A
        )
        
        # Compute channels
        h_R_A = simulator.channel_model.compute_ris_to_user_channel(simulator.user_A)
        h_R_B = simulator.channel_model.compute_ris_to_user_channel(simulator.user_B)
        
        H_A = simulator.channel_model.compute_cascaded_channel(
            h_R_A, simulator.G, phase_shifts, mask
        )
        H_B = simulator.channel_model.compute_cascaded_channel(
            h_R_B, simulator.G, phase_shifts, mask
        )
        
        H_users = np.column_stack([H_A, H_B])
        W = simulator._compute_precoder(H_users)
        
        sinr_A = simulator.compute_sinr(H_users, W, 0)
        sinr_B = simulator.compute_sinr(H_users, W, 1)
        
        sinr_data['User A'].append(sinr_A)
        sinr_data['User B'].append(sinr_B)
    
    return np.array(aperture_sizes), sinr_data


def main():
    # Load config
    config_dict = load_config('configs/system_config.yaml')
    config = SystemConfig(**config_dict)
    
    print(config.get_info())
    
    # Create results directory
    results_dir = Path('results/plots')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Sum-Rate vs. Distance
    print("\n=== Evaluating Sum-Rate vs. Distance ===")
    distances = np.linspace(2, 25, 20)
    results = evaluate_snapshot_a(config, distances)
    
    sum_rates = {k: v['sum_rates'] for k, v in results.items()}
    plot_sum_rate_vs_distance(
        distances,
        sum_rates,
        save_path=str(results_dir / 'sum_rate_vs_distance.png')
    )
    
    # 2. Fairness vs. Distance
    print("\n=== Plotting Fairness vs. Distance ===")
    fairness = {k: v['fairness'] for k, v in results.items()}
    plot_fairness_vs_distance(
        distances,
        fairness,
        save_path=str(results_dir / 'fairness_vs_distance.png')
    )
    
    # 3. Aperture Size Influence
    print("\n=== Evaluating Aperture Size Influence ===")
    aperture_sizes, sinr_data = evaluate_aperture_influence(config)
    
    # Convert to numpy arrays
    sinr_data = {k: np.array(v) for k, v in sinr_data.items()}
    
    plot_aperture_influence(
        aperture_sizes,
        sinr_data,
        save_path=str(results_dir / 'aperture_influence.png')
    )
    
    print(f"\n✓ All plots saved to {results_dir}")


if __name__ == "__main__":
    main()