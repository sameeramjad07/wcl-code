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


def evaluate_all_baselines(config: SystemConfig, distances: np.ndarray):
    """
    Evaluate ALL baseline methods properly.
    
    FIXED: Include random and exhaustive search
    """
    simulator = SystemSimulator(config)
    optimizer = BaselineOptimizer(simulator)
    
    # Initialize results storage
    results = {
        'Random Search': {'sum_rates': [], 'fairness': []},
        'Near-Field Only': {'sum_rates': [], 'fairness': []},
        'Far-Field Only': {'sum_rates': [], 'fairness': []},
        'Adaptive Threshold': {'sum_rates': [], 'fairness': []},
        'Exhaustive Search': {'sum_rates': [], 'fairness': []},
    }
    
    print("\n" + "="*70)
    print("Evaluating All Baseline Methods")
    print("="*70)
    
    for r_A in tqdm(distances, desc="Testing distances"):
        simulator.set_snapshot_A(r_A)
        
        # Determine appropriate aperture based on regime
        is_near_field = r_A < config.d_FF
        aperture = 1024 if is_near_field else 4096
        
        print(f"\nr_A = {r_A:.1f}m ({'near-field' if is_near_field else 'far-field'})")
        
        # 1. Random Search (WORST - baseline)
        print("  [1/5] Random Search...", end=" ")
        res_random = optimizer.random_search(n_iterations=50, aperture_size=aperture)
        results['Random Search']['sum_rates'].append(res_random['sum_rate'])
        results['Random Search']['fairness'].append(
            simulator.compute_jains_fairness(res_random['rate_A'], res_random['rate_B'])
        )
        print(f"Sum-Rate: {res_random['sum_rate']:.3f} bps/Hz")
        
        # 2. Near-Field Only
        print("  [2/5] Near-Field Only...", end=" ")
        res_nf = optimizer.near_field_baseline(aperture_size=1024)
        results['Near-Field Only']['sum_rates'].append(res_nf['sum_rate'])
        results['Near-Field Only']['fairness'].append(
            simulator.compute_jains_fairness(res_nf['rate_A'], res_nf['rate_B'])
        )
        print(f"Sum-Rate: {res_nf['sum_rate']:.3f} bps/Hz")
        
        # 3. Far-Field Only
        print("  [3/5] Far-Field Only...", end=" ")
        res_ff = optimizer.far_field_baseline(aperture_size=4096)
        results['Far-Field Only']['sum_rates'].append(res_ff['sum_rate'])
        results['Far-Field Only']['fairness'].append(
            simulator.compute_jains_fairness(res_ff['rate_A'], res_ff['rate_B'])
        )
        print(f"Sum-Rate: {res_ff['sum_rate']:.3f} bps/Hz")
        
        # 4. Adaptive Threshold
        print("  [4/5] Adaptive Threshold...", end=" ")
        res_adaptive = optimizer.adaptive_threshold_baseline(aperture_size=aperture)
        results['Adaptive Threshold']['sum_rates'].append(res_adaptive['sum_rate'])
        results['Adaptive Threshold']['fairness'].append(
            simulator.compute_jains_fairness(res_adaptive['rate_A'], res_adaptive['rate_B'])
        )
        print(f"Sum-Rate: {res_adaptive['sum_rate']:.3f} bps/Hz")
        
        # 5. Exhaustive Search (BEST - upper bound)
        print("  [5/5] Exhaustive Search...", end=" ")
        res_exhaustive = optimizer.exhaustive_search(
            n_phase_levels=16, 
            aperture_size=aperture, 
            n_samples=1000
        )
        results['Exhaustive Search']['sum_rates'].append(res_exhaustive['sum_rate'])
        results['Exhaustive Search']['fairness'].append(
            simulator.compute_jains_fairness(res_exhaustive['rate_A'], res_exhaustive['rate_B'])
        )
        print(f"Sum-Rate: {res_exhaustive['sum_rate']:.3f} bps/Hz")
    
    return results


def main():
    # Load config
    config = SystemConfig()
    
    print(config.get_info())
    
    # Create results directory
    results_dir = Path('results/plots')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Test distances (focus on transition region)
    distances = np.array([2, 3, 5, 7, 10, 12, 15, 18, 20, 22, 25])
    
    print(f"\nFar-field boundary: {config.d_FF:.2f}m")
    print(f"Testing {len(distances)} distance points")
    
    # Evaluate all baselines
    results = evaluate_all_baselines(config, distances)
    
    # Generate plots
    print("\n" + "="*70)
    print("Generating Plots")
    print("="*70)
    
    # 1. Sum-Rate vs. Distance
    print("\n[1/3] Sum-Rate vs Distance...")
    sum_rates = {k: v['sum_rates'] for k, v in results.items()}
    plot_sum_rate_vs_distance(
        distances,
        sum_rates,
        save_path=str(results_dir / 'sum_rate_vs_distance.png')
    )
    
    # 2. Fairness vs. Distance
    print("[2/3] Fairness vs Distance...")
    fairness = {k: v['fairness'] for k, v in results.items()}
    plot_fairness_vs_distance(
        distances,
        fairness,
        save_path=str(results_dir / 'fairness_vs_distance.png')
    )
    
    # 3. Aperture Size Influence
    print("[3/3] Aperture Influence...")
    aperture_sizes, sinr_data = evaluate_aperture_influence(config)
    sinr_data_arrays = {k: np.array(v) for k, v in sinr_data.items()}
    plot_aperture_influence(
        aperture_sizes,
        sinr_data_arrays,
        save_path=str(results_dir / 'aperture_influence.png')
    )
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    for method_name, method_results in results.items():
        avg_sum_rate = np.mean(method_results['sum_rates'])
        avg_fairness = np.mean(method_results['fairness'])
        print(f"{method_name:20s}: Avg Sum-Rate = {avg_sum_rate:.3f} bps/Hz, "
              f"Avg Fairness = {avg_fairness:.3f}")
    
    print("\n" + "="*70)
    print(f"✓ All plots saved to {results_dir}")
    print("="*70)


def evaluate_aperture_influence(config: SystemConfig):
    """Evaluate aperture size influence on SINR."""
    simulator = SystemSimulator(config)
    simulator.set_snapshot_A(r_A=2.0)  # User A very close
    
    aperture_sizes = [256, 576, 1024, 2304, 4096]  # 16x16, 24x24, 32x32, 48x48, 64x64
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
        
        # Normalize
        H_A = H_A / (np.linalg.norm(H_A) + 1e-10)
        H_B = H_B / (np.linalg.norm(H_B) + 1e-10)
        
        H_users = np.column_stack([H_A, H_B])
        W = simulator._compute_precoder(H_users)
        
        sinr_A = simulator.compute_sinr(H_users, W, 0)
        sinr_B = simulator.compute_sinr(H_users, W, 1)
        
        sinr_data['User A'].append(sinr_A)
        sinr_data['User B'].append(sinr_B)
    
    return np.array(aperture_sizes), sinr_data


if __name__ == "__main__":
    main()