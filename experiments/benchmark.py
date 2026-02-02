"""
Comprehensive Benchmark
========================

Compare all methods and generate complete results.
"""

import sys
sys.path.append('.')

import numpy as np
from typing import Dict, Optional
from pathlib import Path
import json
from tqdm import tqdm

from src.core.system_model import SystemConfig, SystemSimulator, BaselineOptimizer
from src.utils.visualization import (
    plot_sum_rate_vs_distance,
    plot_fairness_vs_distance,
    plot_convergence
)
from src.utils.config import load_config


def exhaustive_search_quantized(
    simulator: SystemSimulator,
    n_phase_levels: int = 8,
    aperture_size: Optional[int] = None,
    max_samples: int = 1000
) -> Dict:
    """
    Exhaustive search with quantized phase shifts (computationally feasible).
    
    Args:
        simulator: System simulator
        n_phase_levels: Number of quantization levels for phases
        aperture_size: Aperture size (None = full)
        max_samples: Maximum random samples to try
        
    Returns:
        Best configuration found
    """
    best_sum_rate = -np.inf
    best_config = None
    
    # Generate aperture mask
    mask = None
    if aperture_size is not None and aperture_size < simulator.config.N_total:
        mask = simulator.ris_controller.generate_aperture_mask(aperture_size)
        n_active = int(np.sum(mask))
    else:
        n_active = simulator.config.N_total
    
    # Quantized phase levels
    phase_levels = np.linspace(0, 2*np.pi, n_phase_levels, endpoint=False)
    
    # Random sampling from quantized space
    for _ in range(max_samples):
        # Random quantized phases
        phase_indices = np.random.randint(0, n_phase_levels, n_active)
        phase_shifts_active = phase_levels[phase_indices]
        
        # Full phase array
        if mask is not None:
            phase_shifts = np.zeros(simulator.config.N_total)
            active_indices = np.where(mask == 1)[0]
            phase_shifts[active_indices] = phase_shifts_active
        else:
            phase_shifts = phase_shifts_active
        
        # Evaluate
        sum_rate, rate_A, rate_B = simulator.compute_sum_rate(phase_shifts, mask)
        
        if sum_rate > best_sum_rate:
            best_sum_rate = sum_rate
            best_config = {
                'phase_shifts': phase_shifts,
                'mask': mask,
                'sum_rate': sum_rate,
                'rate_A': rate_A,
                'rate_B': rate_B
            }
    
    return best_config

def benchmark_all_methods():
    """Run comprehensive benchmark."""
    
    # Load config
    config = SystemConfig()
    simulator = SystemSimulator(config)
    optimizer = BaselineOptimizer(simulator)
    
    # Test distances
    distances = np.linspace(2, 25, 20)
    
    # Results storage
    results = {
        'Random Search': {'sum_rates': [], 'fairness': [], 'time': []},
        'Near-Field Only': {'sum_rates': [], 'fairness': [], 'time': []},
        'Far-Field Only': {'sum_rates': [], 'fairness': [], 'time': []},
        'Adaptive Threshold': {'sum_rates': [], 'fairness': [], 'time': []},
        'Exhaustive Search': {'sum_rates': [], 'fairness': []},
    }
    
    print("=== Running Comprehensive Benchmark ===\n")
    
    for r_A in tqdm(distances, desc="Testing distances"):
        simulator.set_snapshot_A(r_A)
        
        # Determine appropriate aperture
        is_near = r_A < config.d_FF
        aperture = 1024 if is_near else 4096
        
        # Random Search
        res = optimizer.random_search(n_iterations=100, aperture_size=aperture)
        results['Random Search']['sum_rates'].append(res['sum_rate'])
        results['Random Search']['fairness'].append(
            simulator.compute_jains_fairness(res['rate_A'], res['rate_B'])
        )
        
        # Near-Field Only
        res = optimizer.near_field_baseline(aperture_size=1024)
        results['Near-Field Only']['sum_rates'].append(res['sum_rate'])
        results['Near-Field Only']['fairness'].append(
            simulator.compute_jains_fairness(res['rate_A'], res['rate_B'])
        )
        
        # Far-Field Only
        res = optimizer.far_field_baseline(aperture_size=4096)
        results['Far-Field Only']['sum_rates'].append(res['sum_rate'])
        results['Far-Field Only']['fairness'].append(
            simulator.compute_jains_fairness(res['rate_A'], res['rate_B'])
        )
        
        # Adaptive Threshold
        res = optimizer.adaptive_threshold_baseline(aperture_size=aperture)
        results['Adaptive Threshold']['sum_rates'].append(res['sum_rate'])
        results['Adaptive Threshold']['fairness'].append(
            simulator.compute_jains_fairness(res['rate_A'], res['rate_B'])
        )
        
        # Exhaustive Search (NEW)
        res = exhaustive_search_quantized(simulator, n_phase_levels=8, 
                                         aperture_size=aperture, max_samples=500)
        results['Exhaustive Search']['sum_rates'].append(res['sum_rate'])
        results['Exhaustive Search']['fairness'].append(
            simulator.compute_jains_fairness(res['rate_A'], res['rate_B'])
        )
    
    # Generate plots
    plots_dir = Path('results/plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Sum-rate plot
    sum_rates = {k: v['sum_rates'] for k, v in results.items()}
    plot_sum_rate_vs_distance(
        distances,
        sum_rates,
        save_path=str(plots_dir / 'benchmark_sum_rate.png'),
        title="Benchmark: Sum-Rate vs. Distance"
    )
    
    # Fairness plot
    fairness = {k: v['fairness'] for k, v in results.items()}
    plot_fairness_vs_distance(
        distances,
        fairness,
        save_path=str(plots_dir / 'benchmark_fairness.png'),
        title="Benchmark: Fairness vs. Distance"
    )
    
    # Save numerical results
    with open(plots_dir / 'benchmark_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for method, data in results.items():
            serializable_results[method] = {
                k: [float(x) for x in v] for k, v in data.items()
            }
        json.dump({
            'distances': distances.tolist(),
            'results': serializable_results
        }, f, indent=2)
    
    print(f"\n✓ Benchmark complete! Results saved to {plots_dir}")


if __name__ == "__main__":
    benchmark_all_methods()