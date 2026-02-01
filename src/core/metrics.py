"""
Performance Metrics for XL-RIS System
=====================================

Metrics:
- SINR (Signal-to-Interference-plus-Noise Ratio)
- Sum-rate (bps/Hz)
- Jain's Fairness Index
- Delta SNR

Author: Research Team
Date: January 2026
"""

import numpy as np
from typing import Tuple, Optional


def compute_sinr(H_k: np.ndarray, W: np.ndarray, user_idx: int, noise_power: float) -> float:
    """
    Compute SINR for a specific user.
    
    Args:
        H_k: Cascaded channel for user k (M_antennas,)
        W: Precoding matrix (M_antennas, K_users)
        user_idx: Index of the user
        noise_power: Noise power in linear scale
        
    Returns:
        sinr: SINR in linear scale
    """
    # Signal power: |H_k^H * w_k|^2
    signal = np.abs(H_k.conj().T @ W[:, user_idx])**2
    
    # Interference power: sum_{j!=k} |H_k^H * w_j|^2
    interference = 0.0
    for j in range(W.shape[1]):
        if j != user_idx:
            interference += np.abs(H_k.conj().T @ W[:, j])**2
    
    # SINR
    epsilon = 1e-20
    sinr = signal / (interference + noise_power + epsilon)
    
    return max(sinr, epsilon)


def compute_sum_rate(sinrs: np.ndarray) -> float:
    """
    Compute sum-rate from SINRs.
    
    Args:
        sinrs: Array of SINRs in linear scale
        
    Returns:
        sum_rate: Sum-rate in bps/Hz
    """
    rates = np.log2(1 + sinrs)
    return np.sum(rates)


def compute_jains_fairness(rates: np.ndarray) -> float:
    """
    Compute Jain's Fairness Index.
    
    F = (sum r_k)^2 / (K * sum r_k^2)
    
    Perfect fairness: F = 1
    Complete unfairness: F = 1/K
    
    Args:
        rates: Array of user rates
        
    Returns:
        fairness: Fairness index in [0, 1]
    """
    K = len(rates)
    if K == 0:
        return 0.0
    
    sum_rates = np.sum(rates)
    sum_rates_squared = np.sum(rates**2)
    
    if sum_rates_squared < 1e-10:
        return 0.0
    
    fairness = (sum_rates**2) / (K * sum_rates_squared)
    return fairness


def compute_delta_snr(achieved_snr_dB: float, target_snr_dB: float) -> float:
    """
    Compute delta SNR: achieved SNR - target SNR.
    
    Positive delta means target is exceeded.
    Negative delta means target is not met.
    
    Args:
        achieved_snr_dB: Achieved SNR in dB
        target_snr_dB: Target SNR in dB
        
    Returns:
        delta_snr: Delta SNR in dB
    """
    return achieved_snr_dB - target_snr_dB


def compute_spectral_efficiency(sinr: float) -> float:
    """
    Compute spectral efficiency (Shannon capacity).
    
    Args:
        sinr: SINR in linear scale
        
    Returns:
        efficiency: Spectral efficiency in bps/Hz
    """
    return np.log2(1 + sinr)


def compute_power_efficiency(sum_rate: float, total_power: float) -> float:
    """
    Compute power efficiency (bits/Joule).
    
    Args:
        sum_rate: Sum-rate in bps/Hz
        total_power: Total transmit power in Watts
        
    Returns:
        efficiency: Power efficiency in bits/Joule/Hz
    """
    if total_power < 1e-10:
        return 0.0
    return sum_rate / total_power


def compute_interference_leakage(H_users: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Compute interference leakage for each user.
    
    Args:
        H_users: Cascaded channels (M_antennas, K_users)
        W: Precoding matrix (M_antennas, K_users)
        
    Returns:
        leakage: Interference leakage array (K_users,)
    """
    K = H_users.shape[1]
    leakage = np.zeros(K)
    
    for k in range(K):
        H_k = H_users[:, k]
        for j in range(K):
            if j != k:
                leakage[k] += np.abs(H_k.conj().T @ W[:, j])**2
    
    return leakage