"""
XL-RIS System Model for Agentic-RIS Paper
==========================================

Exact specifications from paper:
- Base Station: M=16 antennas (Uniform Linear Array)
- XL-RIS: N=4,096 elements (64×64 Uniform Planar Array)
- Frequency: 28 GHz
- Channel Model: Spherical Wavefront Model (SWM) for ALL links

Author: Research Team
Date: January 2026
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(__file__))  # Add current directory to Python path
from channel_model import ChannelModel, RISController, User


@dataclass
class SystemConfig:
    """Configuration parameters matching the paper specifications."""
    
    # RIS Configuration (64×64 UPA)
    N_y: int = 64  # Number of RIS elements along y-axis
    N_z: int = 64  # Number of RIS elements along z-axis
    d_element: float = 0.5  # Element spacing in wavelengths (λ/2)
    
    # Base Station Configuration (16-antenna ULA)
    M_antennas: int = 16  # Number of BS antennas
    bs_antenna_spacing: float = 0.5  # BS antenna spacing in wavelengths
    P_max_dBm: float = 30.0  # Maximum transmit power in dBm
    
    # RF Parameters
    freq_GHz: float = 28.0  # Carrier frequency in GHz (mmWave)
    noise_power_dBm: float = -90.0  # Noise power in dBm
    
    # Geometry (distances in meters)
    BS_position: np.ndarray = None  # BS position [x, y, z]
    RIS_position: np.ndarray = None  # RIS center position [x, y, z]
    
    # User requirements for delta SNR calculation
    target_SNR_dB: float = 10.0  # Target SNR for fairness metrics
    
    def __post_init__(self):
        """Initialize derived parameters."""
        # Set default positions if not provided
        if self.BS_position is None:
            self.BS_position = np.array([-10.0, 0.0, 0.0])
        if self.RIS_position is None:
            self.RIS_position = np.array([0.0, 0.0, 0.0])
        
        # Compute derived parameters
        self.N_total = self.N_y * self.N_z  # Total RIS elements (4096)
        self.wavelength = 3e8 / (self.freq_GHz * 1e9)  # Wavelength in meters
        self.element_spacing = self.d_element * self.wavelength  # Actual spacing
        
        # Compute RIS aperture size
        self.D_y = (self.N_y - 1) * self.element_spacing
        self.D_z = (self.N_z - 1) * self.element_spacing
        self.D_max = max(self.D_y, self.D_z)
        
        # Compute far-field boundary (Rayleigh distance)
        self.d_FF = 2 * self.D_max**2 / self.wavelength
        
        # Convert power to linear scale
        self.P_max = 10**(self.P_max_dBm / 10) / 1000  # Watts
        self.noise_power = 10**(self.noise_power_dBm / 10) / 1000  # Watts
        self.target_SNR_linear = 10**(self.target_SNR_dB / 10)
        
    def get_info(self) -> str:
        """Return system configuration information."""
        info = f"""
        ╔══════════════════════════════════════════════════════════╗
        ║         XL-RIS Agentic System Configuration             ║
        ╠══════════════════════════════════════════════════════════╣
        ║ RIS Configuration (64×64 UPA):                          ║
        ║   - Elements: {self.N_y} × {self.N_z} = {self.N_total} total              ║
        ║   - Element spacing: λ/{1/self.d_element:.1f} = {self.element_spacing*1000:.2f} mm              ║
        ║   - Aperture size: {self.D_y:.2f} m × {self.D_z:.2f} m              ║
        ║   - Far-field boundary: {self.d_FF:.2f} m                   ║
        ║                                                          ║
        ║ Base Station (16-antenna ULA):                          ║
        ║   - Antennas: {self.M_antennas}                                      ║
        ║   - Antenna spacing: λ/{1/self.bs_antenna_spacing:.1f}                         ║
        ║   - Position: [{self.BS_position[0]:.1f}, {self.BS_position[1]:.1f}, {self.BS_position[2]:.1f}] m            ║
        ║   - TX Power: {self.P_max_dBm} dBm                                ║
        ║                                                          ║
        ║ RF Parameters:                                           ║
        ║   - Frequency: {self.freq_GHz} GHz                               ║
        ║   - Wavelength: {self.wavelength*1000:.2f} mm                           ║
        ║   - Noise Power: {self.noise_power_dBm} dBm                         ║
        ║                                                          ║
        ║ RIS Position: [{self.RIS_position[0]:.1f}, {self.RIS_position[1]:.1f}, {self.RIS_position[2]:.1f}] m                   ║
        ╚══════════════════════════════════════════════════════════╝
        """
        return info


class SystemSimulator:
    """Main simulator for XL-RIS system with exact paper specifications."""
    
    def __init__(self, config: SystemConfig):
        """Initialize simulator."""
        self.config = config
        self.channel_model = ChannelModel(config)
        self.ris_controller = RISController(self.channel_model)
        
        # Pre-compute BS-to-RIS channel (quasi-static)
        self.G = self.channel_model.compute_bs_to_ris_channel()
        
        # Two users as per paper
        self.user_A: Optional[User] = None
        self.user_B: Optional[User] = None
        
    def set_snapshot_A(self, r_A: float):
        """
        Set Snapshot A: Mixed-Field Transition.
        
        Args:
            r_A: Distance of User A (2-25m)
        """
        self.user_A = User('UA', r=r_A, theta_deg=45, z=0.0)
        self.user_B = User('UB', r=50.0, theta_deg=45, z=0.0)
        
    def set_snapshot_B(self):
        """Set Snapshot B: Far-Field NOMA."""
        self.user_A = User('UA', r=45.0, theta_deg=45, z=2.0)
        self.user_B = User('UB', r=50.0, theta_deg=45, z=0.0)
    
    def compute_sinr(self, H_users: np.ndarray, W: np.ndarray,
                    user_idx: int) -> float:
        """
        Compute SINR for a specific user.
        
        Args:
            H_users: Cascaded channels (M_antennas, 2)
            W: Precoding matrix (M_antennas, 2)
            user_idx: 0 for UA, 1 for UB
            
        Returns:
            sinr: SINR in linear scale
        """
        # Get user's channel
        H_k = H_users[:, user_idx]
        
        # Signal power: |H_k^H * w_k|^2
        signal = np.abs(H_k.conj().T @ W[:, user_idx])**2
        
        # Interference power: |H_k^H * w_j|^2 where j != k
        interference = np.abs(H_k.conj().T @ W[:, 1-user_idx])**2
        
        # SINR with small epsilon to avoid division issues
        epsilon = 1e-20
        sinr = signal / (interference + self.config.noise_power + epsilon)
        
        return max(sinr, epsilon)  # Ensure positive
    
    def compute_sum_rate(self, phase_shifts: np.ndarray,
                        aperture_mask: Optional[np.ndarray] = None,
                        power_allocation: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
        """
        Compute sum-rate and individual rates.
        
        Args:
            phase_shifts: RIS phase shifts (N_total,)
            aperture_mask: Binary mask (N_total,)
            power_allocation: Power allocation [p_A, p_B], default equal
            
        Returns:
            (sum_rate, rate_A, rate_B): Rates in bps/Hz
        """
        if self.user_A is None or self.user_B is None:
            raise ValueError("Users not set. Call set_snapshot_A() or set_snapshot_B() first.")
        
        # Compute RIS-to-user channels
        h_R_A = self.channel_model.compute_ris_to_user_channel(self.user_A)
        h_R_B = self.channel_model.compute_ris_to_user_channel(self.user_B)
        
        # Compute cascaded channels
        H_A = self.channel_model.compute_cascaded_channel(h_R_A, self.G, phase_shifts, aperture_mask)
        H_B = self.channel_model.compute_cascaded_channel(h_R_B, self.G, phase_shifts, aperture_mask)
        
        H_users = np.column_stack([H_A, H_B])
        
        # Compute precoder (MRT with power allocation)
        W = self._compute_precoder(H_users, power_allocation)
        
        # Compute SINRs
        sinr_A = self.compute_sinr(H_users, W, 0)
        sinr_B = self.compute_sinr(H_users, W, 1)
        
        # Compute rates
        rate_A = np.log2(1 + sinr_A)
        rate_B = np.log2(1 + sinr_B)
        sum_rate = rate_A + rate_B
        
        return sum_rate, rate_A, rate_B
    
    def compute_delta_snr(self, phase_shifts: np.ndarray,
                         aperture_mask: Optional[np.ndarray] = None,
                         power_allocation: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """
        Compute delta SNR: achieved SNR - target SNR.
        
        For fairness metrics and QoS evaluation.
        
        Returns:
            (delta_A, delta_B): Delta SNR in dB for both users
        """
        # Get channels
        h_R_A = self.channel_model.compute_ris_to_user_channel(self.user_A)
        h_R_B = self.channel_model.compute_ris_to_user_channel(self.user_B)
        
        H_A = self.channel_model.compute_cascaded_channel(h_R_A, self.G, phase_shifts, aperture_mask)
        H_B = self.channel_model.compute_cascaded_channel(h_R_B, self.G, phase_shifts, aperture_mask)
        
        H_users = np.column_stack([H_A, H_B])
        W = self._compute_precoder(H_users, power_allocation)
        
        # Compute achieved SNR (ignoring interference for delta SNR)
        snr_A = np.abs(H_A.conj() @ W[:, 0])**2 / self.config.noise_power
        snr_B = np.abs(H_B.conj() @ W[:, 1])**2 / self.config.noise_power
        
        # Delta SNR in dB
        delta_A = 10 * np.log10(snr_A) - self.config.target_SNR_dB
        delta_B = 10 * np.log10(snr_B) - self.config.target_SNR_dB
        
        return delta_A, delta_B
    
    def _compute_precoder(self, H: np.ndarray,
                         power_allocation: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute MRT precoder with power allocation.
        
        Args:
            H: Channel matrix (M_antennas, 2)
            power_allocation: [p_A, p_B] where p_A + p_B = 1
            
        Returns:
            W: Precoding matrix (M_antennas, 2)
        """
        # Default: equal power allocation
        if power_allocation is None:
            power_allocation = np.array([0.5, 0.5])
        
        # MRT: w_k = H_k^* / ||H_k||
        W = H.conj()
        
        # Normalize each column
        for k in range(2):
            norm = np.linalg.norm(W[:, k])
            if norm > 1e-10:
                W[:, k] = W[:, k] / norm
            # Apply power allocation
            W[:, k] = W[:, k] * np.sqrt(power_allocation[k] * self.config.P_max)
        
        return W
    
    def compute_jains_fairness(self, rate_A: float, rate_B: float) -> float:
        """
        Compute Jain's Fairness Index.
        
        F = (sum r_k)^2 / (K * sum r_k^2)
        
        Perfect fairness: F = 1
        """
        rates = np.array([rate_A, rate_B])
        fairness = (np.sum(rates)**2) / (2 * np.sum(rates**2))
        return fairness


# ============================================================================
# Baseline Optimization Methods
# ============================================================================

class BaselineOptimizer:
    """Baseline methods for comparison."""
    
    def __init__(self, simulator: SystemSimulator):
        """Initialize optimizer."""
        self.simulator = simulator
        self.ris_controller = simulator.ris_controller
    
    def random_search(self, n_iterations: int = 100,
                     aperture_size: Optional[int] = None) -> Dict:
        """Random search baseline."""
        best_sum_rate = -np.inf
        best_config = None
        
        # Generate aperture mask if specified
        mask = None
        if aperture_size is not None:
            mask = self.ris_controller.generate_aperture_mask(aperture_size)
        
        for i in range(n_iterations):
            # Random phase shifts
            phase_shifts = np.random.uniform(0, 2*np.pi, self.simulator.config.N_total)
            
            # Evaluate
            sum_rate, rate_A, rate_B = self.simulator.compute_sum_rate(phase_shifts, mask)
            
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
    
    def near_field_baseline(self, aperture_size: Optional[int] = None) -> Dict:
        """Always use near-field beamfocusing."""
        # Focus on User A (primary user)
        phase_shifts = self.ris_controller.compute_near_field_phase_profile(self.simulator.user_A)
        
        mask = None
        if aperture_size is not None:
            mask = self.ris_controller.generate_aperture_mask(aperture_size)
        
        sum_rate, rate_A, rate_B = self.simulator.compute_sum_rate(phase_shifts, mask)
        
        return {
            'phase_shifts': phase_shifts,
            'mask': mask,
            'sum_rate': sum_rate,
            'rate_A': rate_A,
            'rate_B': rate_B
        }
    
    def far_field_baseline(self, aperture_size: Optional[int] = None) -> Dict:
        """Always use far-field beamsteering."""
        # Steer toward User A
        phase_shifts = self.ris_controller.compute_far_field_phase_profile(self.simulator.user_A)
        
        mask = None
        if aperture_size is not None:
            mask = self.ris_controller.generate_aperture_mask(aperture_size)
        
        sum_rate, rate_A, rate_B = self.simulator.compute_sum_rate(phase_shifts, mask)
        
        return {
            'phase_shifts': phase_shifts,
            'mask': mask,
            'sum_rate': sum_rate,
            'rate_A': rate_A,
            'rate_B': rate_B
        }
    
    def adaptive_threshold_baseline(self, aperture_size: Optional[int] = None) -> Dict:
        """
        Threshold-based regime selection.
        
        Uses near-field if UA < d_FF, far-field otherwise.
        """
        is_near_field = self.simulator.user_A.is_near_field(
            self.simulator.config.d_FF, self.simulator.config.RIS_position
        )
        
        if is_near_field:
            return self.near_field_baseline(aperture_size)
        else:
            return self.far_field_baseline(aperture_size)