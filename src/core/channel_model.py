"""
Channel Model and RIS Control for XL-RIS System
================================================

Implements:
- Spherical Wavefront Model (SWM) for all links
- RIS phase shift controller
- User positioning and tracking

Author: Research Team
Date: January 2026
"""

import numpy as np
from typing import Tuple, Optional


class User:
    """Represents a user with spherical coordinates relative to RIS."""
    
    def __init__(self, user_id: str, r: float, theta_deg: float, z: float = 0.0):
        """
        Initialize user with spherical coordinates.
        
        Args:
            user_id: Unique identifier
            r: Distance from RIS in meters
            theta_deg: Azimuth angle in degrees
            z: Height in meters (default 0)
        """
        self.user_id = user_id
        self.r = r
        self.theta = np.deg2rad(theta_deg)
        self.z = z
        
        # Convert to Cartesian coordinates
        self.position = self._spherical_to_cartesian()
        
    def _spherical_to_cartesian(self) -> np.ndarray:
        """Convert spherical to Cartesian coordinates."""
        x = self.r * np.cos(self.theta)
        y = self.r * np.sin(self.theta)
        z = self.z
        return np.array([x, y, z])
    
    def update_position(self, r: Optional[float] = None, 
                       theta_deg: Optional[float] = None,
                       z: Optional[float] = None):
        """Update user position."""
        if r is not None:
            self.r = r
        if theta_deg is not None:
            self.theta = np.deg2rad(theta_deg)
        if z is not None:
            self.z = z
        self.position = self._spherical_to_cartesian()
    
    def get_distance_to_ris(self, ris_position: np.ndarray) -> float:
        """Compute distance from user to RIS center."""
        return np.linalg.norm(self.position - ris_position)
    
    def is_near_field(self, d_FF: float, ris_position: np.ndarray) -> bool:
        """Check if user is in near-field region."""
        return self.get_distance_to_ris(ris_position) < d_FF


class ChannelModel:
    """Spherical Wavefront Model (SWM) for all links - CRITICAL for DDM."""
    
    def __init__(self, config):
        """Initialize channel model with system configuration."""
        self.config = config
        self._generate_ris_element_positions()
        self._generate_bs_antenna_positions()
        
    def _generate_ris_element_positions(self):
        """Generate 3D positions of all RIS elements (64×64 UPA)."""
        y_coords = np.linspace(
            -self.config.D_y/2, self.config.D_y/2, self.config.N_y
        )
        z_coords = np.linspace(
            -self.config.D_z/2, self.config.D_z/2, self.config.N_z
        )
        
        # Create mesh grid
        Y, Z = np.meshgrid(y_coords, z_coords)
        
        # Flatten and create position matrix (4096 × 3)
        self.element_positions = np.zeros((self.config.N_total, 3))
        self.element_positions[:, 0] = self.config.RIS_position[0]  # x = 0
        self.element_positions[:, 1] = Y.flatten() + self.config.RIS_position[1]
        self.element_positions[:, 2] = Z.flatten() + self.config.RIS_position[2]
        
        # Store grid indices for visualization
        self.element_grid_indices = np.array([
            [i, j] for i in range(self.config.N_z) for j in range(self.config.N_y)
        ])
    
    def _generate_bs_antenna_positions(self):
        """Generate positions of BS antennas (16-antenna ULA along y-axis)."""
        y_coords = np.linspace(
            -(self.config.M_antennas - 1) * self.config.bs_antenna_spacing * self.config.wavelength / 2,
            (self.config.M_antennas - 1) * self.config.bs_antenna_spacing * self.config.wavelength / 2,
            self.config.M_antennas
        )
        
        self.bs_antenna_positions = np.zeros((self.config.M_antennas, 3))
        self.bs_antenna_positions[:, 0] = self.config.BS_position[0]
        self.bs_antenna_positions[:, 1] = y_coords + self.config.BS_position[1]
        self.bs_antenna_positions[:, 2] = self.config.BS_position[2]
    
    def compute_swm_channel(self, source_positions: np.ndarray, 
                        target_position: np.ndarray,
                        is_near_field: bool = True) -> np.ndarray:
        """
        Compute Spherical Wavefront Model (SWM) channel.
        
        CRITICAL: This is essential for DDM - planar wave model cannot simulate
        depth-division multiplexing.
        
        Args:
            source_positions: Source antenna/element positions (N_source × 3)
            target_position: Target position (3,)
            is_near_field: If True, use near-field model with 1/r² path loss
            
        Returns:
            h: Channel vector (N_source,) complex
        """
        # Compute distances from each source to target
        delta = source_positions - target_position
        distances = np.linalg.norm(delta, axis=1)
        
        # Adding small epsilon to avoid division by zero
        epsilon = 1e-10
        
        # Near-field: 1/r² path loss (power) = 1/r amplitude with correction
        # Far-field: 1/r path loss (Friis equation)
        if is_near_field:
            # Near-field model: amplitude ~ 1/r, but we apply sqrt for power normalization
            # Path loss in amplitude: sqrt(λ/(4πr)²) with Rayleigh distance correction
            path_loss_amplitude = self.config.wavelength / (4 * np.pi * (distances + epsilon))
            # Apply near-field correction factor
            correction = np.sqrt(1 / (1 + (self.config.d_FF / (distances + epsilon))**2))
            path_loss_amplitude = path_loss_amplitude * correction
        else:
            # Far-field Friis equation
            path_loss_amplitude = self.config.wavelength / (4 * np.pi * (distances + epsilon))
        
        # Phase based on distance: exp(-j * 2π * r / λ)
        phase = -2 * np.pi * distances / self.config.wavelength
        
        # SWM channel
        h = path_loss_amplitude * np.exp(1j * phase)
        
        return h
    
    def compute_bs_to_ris_channel(self) -> np.ndarray:
        """
        Compute BS-to-RIS channel matrix G using SWM.
        
        FIXED: Proper Rayleigh fading and normalization
        """
        G = np.zeros((self.config.N_total, self.config.M_antennas), dtype=complex)
        
        for m in range(self.config.M_antennas):
            bs_antenna_pos = self.bs_antenna_positions[m, :]
            
            # Distances
            delta = self.element_positions - bs_antenna_pos
            distances = np.linalg.norm(delta, axis=1)
            
            # Path loss
            path_loss = self.config.wavelength / (4 * np.pi * (distances + 1e-10))
            
            # Phase
            phase = -2 * np.pi * distances / self.config.wavelength
            
            # Rayleigh fading (complex Gaussian)
            rayleigh_real = np.random.randn(self.config.N_total)
            rayleigh_imag = np.random.randn(self.config.N_total)
            rayleigh = (rayleigh_real + 1j * rayleigh_imag) / np.sqrt(2)
            
            # Channel
            G[:, m] = path_loss * np.exp(1j * phase) * rayleigh
        
        # Normalize entire matrix
        norm_factor = np.sqrt(np.sum(np.abs(G)**2) / (self.config.N_total * self.config.M_antennas))
        G = G / (norm_factor + 1e-10)
        
        return G
    
    def compute_ris_to_user_channel(self, user: User) -> np.ndarray:
        """
        Compute RIS-to-user channel using SWM.
        
        FIXED: Proper path loss model
        """
        # Compute distances
        delta = self.element_positions - user.position
        distances = np.linalg.norm(delta, axis=1)
        
        # Small epsilon to avoid division by zero
        epsilon = 1e-10
        
        # Check if near-field
        is_near_field = user.is_near_field(self.config.d_FF, self.config.RIS_position)
        
        if is_near_field:
            # Near-field: More accurate spherical wave model
            # Path loss includes both distance and focusing effects
            path_loss_amplitude = self.config.wavelength / (4 * np.pi * (distances + epsilon))
            
            # Apply near-field correction
            d_ratio = distances / (self.config.d_FF + epsilon)
            correction = 1 / np.sqrt(1 + d_ratio**2)
            path_loss_amplitude = path_loss_amplitude * correction
        else:
            # Far-field: Standard Friis equation
            path_loss_amplitude = self.config.wavelength / (4 * np.pi * (distances + epsilon))
        
        # Phase term
        phase = -2 * np.pi * distances / self.config.wavelength
        
        # Channel vector
        h_R = path_loss_amplitude * np.exp(1j * phase)
        
        # Normalize by sqrt(N) for power normalization
        h_R = h_R * np.sqrt(self.config.N_total) / np.sqrt(np.sum(np.abs(h_R)**2) + epsilon)
        
        return h_R
    
    def compute_cascaded_channel(self, h_R: np.ndarray, G: np.ndarray,
                                 phase_shifts: np.ndarray,
                                 aperture_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute cascaded channel: H_k^H = h_R,k^H * Φ_eff * G
        
        Args:
            h_R: RIS-to-user channel (N_total,)
            G: BS-to-RIS channel (N_total, M_antennas)
            phase_shifts: RIS phase shifts (N_total,) in radians
            aperture_mask: Binary mask (N_total,), 1=active, 0=inactive
            
        Returns:
            H_k: Cascaded channel (M_antennas,) complex
        """
        # Create effective reflection coefficients
        phi_eff = np.exp(1j * phase_shifts)
        
        # Apply aperture mask if provided
        if aperture_mask is not None:
            phi_eff = phi_eff * aperture_mask
        
        # Cascaded channel: h_R^H * diag(phi) * G
        reflected = np.conj(phi_eff) * h_R
        H_k = G.T.conj() @ reflected
        
        return H_k


class RISController:
    """Controls RIS phase shifts for beamforming/beamfocusing."""
    
    def __init__(self, channel_model: ChannelModel):
        """Initialize RIS controller."""
        self.channel_model = channel_model
        self.config = channel_model.config
        
    def compute_near_field_phase_profile(self, user: User,
                                         include_bs_compensation: bool = True) -> np.ndarray:
        """
        Compute near-field beamfocusing phase profile for DDM.
        
        This creates a CONVERGING spherical wavefront that focuses energy at the
        user's exact 3D location, enabling depth-division multiplexing.
        
        Args:
            user: Target user
            include_bs_compensation: Compensate for BS-RIS phase
            
        Returns:
            phase_shifts: Phase shifts (N_total,) in [0, 2π]
        """
        # Distance from each RIS element to user
        delta = self.channel_model.element_positions - user.position
        r_n_user = np.linalg.norm(delta, axis=1)
        
        # Phase profile for beamfocusing
        phase_shifts = 2 * np.pi * r_n_user / self.config.wavelength
        
        # Optionally add BS-to-RIS compensation
        if include_bs_compensation:
            # Use center BS antenna for simplicity
            center_antenna_idx = self.config.M_antennas // 2
            bs_pos = self.channel_model.bs_antenna_positions[center_antenna_idx, :]
            delta_bs = self.channel_model.element_positions - bs_pos
            r_n_bs = np.linalg.norm(delta_bs, axis=1)
            phase_shifts += 2 * np.pi * r_n_bs / self.config.wavelength
        
        # Wrap to [0, 2π]
        phase_shifts = np.mod(phase_shifts, 2 * np.pi)
        
        return phase_shifts
    
    def compute_far_field_phase_profile(self, user: User) -> np.ndarray:
        """
        Compute far-field beamsteering phase profile.
        
        For far-field users, creates a planar wavefront in the user's direction.
        
        Args:
            user: Target user
            
        Returns:
            phase_shifts: Phase shifts (N_total,) in [0, 2π]
        """
        # Direction to user
        direction = user.position - self.config.RIS_position
        direction = direction / np.linalg.norm(direction)
        
        # Phase shifts for planar wavefront
        phase_shifts = 2 * np.pi / self.config.wavelength * (
            self.channel_model.element_positions @ direction
        )
        
        # Wrap to [0, 2π]
        phase_shifts = np.mod(phase_shifts, 2 * np.pi)
        
        return phase_shifts
    
    def generate_aperture_mask(self, aperture_size: int) -> np.ndarray:
        """
        Generate aperture mask for partial RIS activation.
        
        This implements "aperture tapering" for interference reduction.
        
        Args:
            aperture_size: Number of active elements (e.g., 16×16=256, 32×32=1024)
            
        Returns:
            mask: Binary array (N_total,)
        """
        if aperture_size >= self.config.N_total:
            return np.ones(self.config.N_total)
        
        # Calculate sub-array size
        n_sub = int(np.sqrt(aperture_size))
        n_sub = min(n_sub, self.config.N_y, self.config.N_z)
        
        # Create center mask
        mask = np.zeros(self.config.N_total)
        center_y = self.config.N_y // 2
        center_z = self.config.N_z // 2
        start_y = center_y - n_sub // 2
        start_z = center_z - n_sub // 2
        
        for i in range(n_sub):
            for j in range(n_sub):
                idx = (start_z + i) * self.config.N_y + (start_y + j)
                if 0 <= idx < self.config.N_total:
                    mask[idx] = 1
        
        return mask