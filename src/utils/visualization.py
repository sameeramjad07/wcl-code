"""
Visualization Functions
=======================

Generate all plots for the paper.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12


def plot_sum_rate_vs_distance(
    distances: np.ndarray,
    results: Dict[str, List[float]],
    save_path: str = None,
    title: str = "Sum-Rate vs. User Distance (Snapshot A)"
):
    """
    Plot sum-rate vs. distance for different methods.
    
    Args:
        distances: Array of UA distances
        results: Dict mapping method name to list of sum-rates
        save_path: Path to save figure
        title: Plot title
    """
    plt.figure(figsize=(12, 7))
    
    markers = ['o', 's', '^', 'D', 'v', '*', 'p']
    colors = sns.color_palette("husl", len(results))
    
    for idx, (method, rates) in enumerate(results.items()):
        plt.plot(
            distances,
            rates,
            marker=markers[idx % len(markers)],
            linewidth=2.5,
            markersize=8,
            label=method,
            color=colors[idx],
            alpha=0.8
        )
    
    plt.xlabel('User A Distance (m)', fontsize=14)
    plt.ylabel('Sum-Rate (bps/Hz)', fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def plot_fairness_vs_distance(
    distances: np.ndarray,
    results: Dict[str, List[float]],
    save_path: str = None,
    title: str = "Jain's Fairness Index vs. User Distance"
):
    """
    Plot fairness index vs. distance.
    
    Args:
        distances: Array of UA distances
        results: Dict mapping method name to list of fairness indices
        save_path: Path to save figure
        title: Plot title
    """
    plt.figure(figsize=(12, 7))
    
    markers = ['o', 's', '^', 'D', 'v', '*', 'p']
    colors = sns.color_palette("Set2", len(results))
    
    for idx, (method, fairness) in enumerate(results.items()):
        plt.plot(
            distances,
            fairness,
            marker=markers[idx % len(markers)],
            linewidth=2.5,
            markersize=8,
            label=method,
            color=colors[idx],
            alpha=0.8
        )
    
    # Add perfect fairness line
    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=2, 
                label='Perfect Fairness', alpha=0.6)
    
    plt.xlabel('User A Distance (m)', fontsize=14)
    plt.ylabel("Jain's Fairness Index", fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.ylim(0.5, 1.05)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def plot_convergence(
    results: Dict[str, Dict[str, List[float]]],
    save_path: str = None,
    title: str = "Training Convergence Comparison"
):
    """
    Plot convergence curves for different DRL methods.
    
    Args:
        results: Dict mapping method to {'episodes': [...], 'rewards': [...]}
        save_path: Path to save figure
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = sns.color_palette("Dark2", len(results))
    
    # Plot raw rewards
    for idx, (method, data) in enumerate(results.items()):
        episodes = data['episodes']
        rewards = data['rewards']
        ax1.plot(episodes, rewards, label=method, color=colors[idx], 
                alpha=0.7, linewidth=2)
    
    ax1.set_xlabel('Episode', fontsize=14)
    ax1.set_ylabel('Reward', fontsize=14)
    ax1.set_title('Episode Reward', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot moving average
    window = 50
    for idx, (method, data) in enumerate(results.items()):
        episodes = data['episodes']
        rewards = data['rewards']
        
        # Compute moving average
        if len(rewards) >= window:
            ma_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ma_episodes = episodes[window-1:]
            ax2.plot(ma_episodes, ma_rewards, label=method, color=colors[idx],
                    linewidth=2.5)
    
    ax2.set_xlabel('Episode', fontsize=14)
    ax2.set_ylabel(f'Moving Average Reward (window={window})', fontsize=14)
    ax2.set_title('Smoothed Convergence', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def plot_aperture_influence(
    aperture_sizes: np.ndarray,
    sinr_data: Dict[str, np.ndarray],
    save_path: str = None,
    title: str = "Aperture Size Influence on SINR (UA at 2m)"
):
    """
    Plot SINR vs. aperture size.
    
    Args:
        aperture_sizes: Array of aperture sizes
        sinr_data: Dict mapping user to SINR array
        save_path: Path to save figure
        title: Plot title
    """
    plt.figure(figsize=(12, 7))
    
    colors = ['#1f77b4', '#ff7f0e']
    markers = ['o', 's']
    
    for idx, (user, sinrs) in enumerate(sinr_data.items()):
        plt.plot(
            aperture_sizes,
            10 * np.log10(sinrs),  # Convert to dB
            marker=markers[idx],
            linewidth=2.5,
            markersize=8,
            label=user,
            color=colors[idx],
            alpha=0.8
        )
    
    plt.xlabel('Number of Active RIS Elements', fontsize=14)
    plt.ylabel('SINR (dB)', fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def plot_system_diagram(config, save_path: str = None):
    """
    Create system geometry visualization.
    
    Args:
        config: System configuration
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot BS
    ax.scatter(*config.BS_position, c='red', marker='^', s=200, label='Base Station')
    
    # Plot RIS
    ax.scatter(*config.RIS_position, c='blue', marker='s', s=300, label='XL-RIS', alpha=0.6)
    
    # Plot example users
    user_positions = [
        [5*np.cos(np.pi/4), 5*np.sin(np.pi/4), 0],
        [50*np.cos(np.pi/4), 50*np.sin(np.pi/4), 0]
    ]
    
    for i, pos in enumerate(user_positions):
        ax.scatter(*pos, c='green', marker='o', s=150, label=f'User {i+1}' if i == 0 else '')
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title('XL-RIS System Geometry', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()