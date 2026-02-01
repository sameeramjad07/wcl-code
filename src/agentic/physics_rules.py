"""
Physics Rules for RIS Optimization
===================================

Electromagnetic physics principles for LLM reasoning.

Author: Research Team
Date: January 2026
"""

from typing import Dict, List

class PhysicsRules:

    def __init__(self):
        pass
    def get_default_physics_rules() -> List[Dict]:
        """
        Return default set of physics rules for RAG database.
        
        Returns:
            List of rule dictionaries
        """
        rules = [
            {
                "title": "Near-Field Beamfocusing",
                "condition": "User distance < 10m (r < 0.5 * d_FF)",
                "action": "Use sub-aperture (N_sub=1024) with near-field phase profile. Priority: Depth-focusing for DDM.",
                "priority": "HIGH",
                "aperture_size": 1024,
                "strategy": "near_field",
                "reward_weights": {"alpha": 1.0, "beta": 1.0, "gamma": 0.5}
            },
            {
                "title": "Transition Region",
                "condition": "User distance 10-20m (0.5*d_FF < r < d_FF)",
                "action": "Use medium aperture (N_sub=2048) with adaptive phase profile. Balance between focusing and steering.",
                "priority": "MEDIUM",
                "aperture_size": 2048,
                "strategy": "adaptive",
                "reward_weights": {"alpha": 1.0, "beta": 1.0, "gamma": 0.3}
            },
            {
                "title": "Far-Field Beamsteering",
                "condition": "User distance > 20m (r > d_FF)",
                "action": "Use full aperture (N_sub=4096) with far-field phase profile. Priority: Maximum array gain.",
                "priority": "HIGH",
                "aperture_size": 4096,
                "strategy": "far_field",
                "reward_weights": {"alpha": 1.0, "beta": 1.0, "gamma": 0.1}
            },
            {
                "title": "Angular Conflict - Near Field",
                "condition": "Both users near-field AND angular separation < 30°",
                "action": "Increase interference penalty (γ=0.8). Use aperture tapering to reduce sidelobes.",
                "priority": "HIGH",
                "aperture_size": 1024,
                "strategy": "near_field",
                "reward_weights": {"alpha": 0.8, "beta": 0.8, "gamma": 0.8}
            },
            {
                "title": "Angular Conflict - Far Field",
                "condition": "Both users far-field AND angular separation < 30°",
                "action": "Switch to NOMA mode. Priority: SIC feasibility (power-domain separation).",
                "priority": "HIGH",
                "aperture_size": 4096,
                "strategy": "noma",
                "reward_weights": {"alpha": 0.6, "beta": 0.4, "gamma": 0.2}
            },
            {
                "title": "Mixed-Field DDM",
                "condition": "One user near-field, one far-field",
                "action": "Exploit depth separation. Use N_sub=2048. Focus on near user while maintaining far-user QoS.",
                "priority": "MEDIUM",
                "aperture_size": 2048,
                "strategy": "mixed_ddm",
                "reward_weights": {"alpha": 1.2, "beta": 0.8, "gamma": 0.4}
            },
            {
                "title": "Height Separation Exploitation",
                "condition": "Vertical separation Δz > 2m",
                "action": "Use 3D beamforming. Adjust elevation pattern to minimize interference.",
                "priority": "MEDIUM",
                "aperture_size": 3072,
                "strategy": "3d_beamforming",
                "reward_weights": {"alpha": 1.0, "beta": 1.0, "gamma": 0.3}
            },
            {
                "title": "Fairness Violation",
                "condition": "Rate ratio > 3:1 OR min_rate < threshold",
                "action": "Increase weight for weaker user (β=1.5). Reduce interference penalty.",
                "priority": "HIGH",
                "aperture_size": None,  # Keep current
                "strategy": "fairness_boost",
                "reward_weights": {"alpha": 0.7, "beta": 1.5, "gamma": 0.1}
            },
            {
                "title": "Power Efficiency Mode",
                "condition": "Sum-rate target achieved",
                "action": "Minimize active elements. Use smallest viable aperture.",
                "priority": "LOW",
                "aperture_size": 1024,
                "strategy": "power_efficient",
                "reward_weights": {"alpha": 0.5, "beta": 0.5, "gamma": 0.05}
            },
            {
                "title": "Interference Mitigation",
                "condition": "Cross-interference > -10dB",
                "action": "Increase γ to 1.0. Use null-steering in interferer direction.",
                "priority": "HIGH",
                "aperture_size": None,
                "strategy": "interference_null",
                "reward_weights": {"alpha": 0.8, "beta": 0.8, "gamma": 1.0}
            },
            {
                "title": "Spherical Wave Dominance",
                "condition": "User within Fresnel region (r < 2D²/λ)",
                "action": "Mandatory near-field model. Planar wave approximation invalid.",
                "priority": "CRITICAL",
                "aperture_size": 1024,
                "strategy": "near_field",
                "reward_weights": {"alpha": 1.0, "beta": 1.0, "gamma": 0.6}
            },
            {
                "title": "Equal Distance Users",
                "condition": "|r_A - r_B| < 2m",
                "action": "Cannot use DDM. Switch to angular/power domain separation.",
                "priority": "HIGH",
                "aperture_size": 4096,
                "strategy": "angular_noma",
                "reward_weights": {"alpha": 0.5, "beta": 0.5, "gamma": 0.3}
            },
            {
                "title": "Extremely Close User",
                "condition": "User distance < 2m",
                "action": "Use minimum aperture (16×16=256). Avoid over-focusing.",
                "priority": "CRITICAL",
                "aperture_size": 256,
                "strategy": "near_field",
                "reward_weights": {"alpha": 1.0, "beta": 1.0, "gamma": 0.9}
            },
            {
                "title": "High Mobility",
                "condition": "User velocity > 5 m/s",
                "action": "Reduce update frequency. Use robust far-field patterns.",
                "priority": "MEDIUM",
                "aperture_size": 4096,
                "strategy": "robust_far_field",
                "reward_weights": {"alpha": 1.0, "beta": 1.0, "gamma": 0.2}
            },
            {
                "title": "QoS Emergency",
                "condition": "Any user rate < 0.5 bps/Hz",
                "action": "Maximize total power to that user. Sacrifice fairness temporarily.",
                "priority": "CRITICAL",
                "aperture_size": 4096,
                "strategy": "emergency",
                "reward_weights": {"alpha": 2.0, "beta": 2.0, "gamma": 0.0}
            }
        ]
        
        return rules


    def analyze_scenario(user_A_distance: float, user_B_distance: float,
                        user_A_angle: float, user_B_angle: float,
                        d_FF: float) -> str:
        """
        Create scenario description for RAG retrieval.
        
        Args:
            user_A_distance: Distance of user A in meters
            user_B_distance: Distance of user B in meters
            user_A_angle: Angle of user A in degrees
            user_B_angle: Angle of user B in degrees
            d_FF: Far-field boundary in meters
            
        Returns:
            scenario: Text description of scenario
        """
        # Determine regimes
        regime_A = "near-field" if user_A_distance < d_FF else "far-field"
        regime_B = "near-field" if user_B_distance < d_FF else "far-field"
        
        # Angular separation
        angular_sep = abs(user_A_angle - user_B_angle)
        
        # Distance separation
        distance_sep = abs(user_A_distance - user_B_distance)
        
        scenario = f"""
    User A: {user_A_distance:.1f}m ({regime_A}), {user_A_angle:.1f}°
    User B: {user_B_distance:.1f}m ({regime_B}), {user_B_angle:.1f}°
    Angular separation: {angular_sep:.1f}°
    Distance separation: {distance_sep:.1f}m
    Far-field boundary: {d_FF:.1f}m
    """
        
        if regime_A == regime_B == "near-field" and angular_sep < 30:
            scenario += "\nCONFLICT: Both users near-field with small angular separation"
        elif regime_A == regime_B == "far-field" and angular_sep < 30:
            scenario += "\nCONFLICT: Both users far-field with small angular separation"
        elif regime_A != regime_B:
            scenario += "\nOPPORTUNITY: Mixed-field scenario - potential for DDM"
        
        if distance_sep < 2:
            scenario += "\nCONSTRAINT: Users at similar distances - DDM difficult"
        
        return scenario.strip()