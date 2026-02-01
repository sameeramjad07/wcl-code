"""
Comprehensive Benchmark
========================

Compare all methods and generate complete results.
"""

import sys
sys.path.append('.')

import numpy as np
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
    }
    
    print("=== Running Comprehensive Benchmark ===\n")
    
    for r_A in tqdm(distances, desc="Testing distances"):
        simulator.set_snapshot_A(r_A)
        
        # Random Search
        res = optimizer.random_search(n_iterations=100)
        results['Random Search']['sum_rates'].append(res['sum_rate'])
        results['Random Search']['fairness'].append(
            simulator.compute_jains_fairness(res['rate_A'], res['rate_B'])
        )
        
        # Near-Field Only
        res = optimizer.near_field_baseline()
        results['Near-Field Only']['sum_rates'].append(res['sum_rate'])
        results['Near-Field Only']['fairness'].append(
            simulator.compute_jains_fairness(res['rate_A'], res['rate_B'])
        )
        
        # Far-Field Only
        res = optimizer.far_field_baseline()
        results['Far-Field Only']['sum_rates'].append(res['sum_rate'])
        results['Far-Field Only']['fairness'].append(
            simulator.compute_jains_fairness(res['rate_A'], res['rate_B'])
        )
        
        # Adaptive Threshold
        res = optimizer.adaptive_threshold_baseline()
        results['Adaptive Threshold']['sum_rates'].append(res['sum_rate'])
        results['Adaptive Threshold']['fairness'].append(
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