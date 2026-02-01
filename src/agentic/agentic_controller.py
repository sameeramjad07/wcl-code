"""
Agentic Controller - Main LLM-DRL Integration
==============================================

Combines LLM reasoning with DRL fine-tuning.

Author: Research Team
Date: January 2026
"""

import numpy as np
from typing import Dict, Tuple, Optional
import json

from src.agentic.llm_interface import LLMInterface
from src.agentic.rag_database import RAGDatabase
from src.agentic.physics_rules import PhysicsRules
from src.agents.sac_agent import SACAgent
from src.core.channel_model import RISController


class AgenticController:
    """Main controller combining LLM and DRL."""
    
    def __init__(
        self,
        llm_interface: LLMInterface,
        rag_database: RAGDatabase,
        ris_controller: RISController,
        physics_rules: PhysicsRules,
        config
    ):
        """
        Initialize agentic controller.
        
        Args:
            llm_interface: LLM interface
            rag_database: RAG database
            ris_controller: RIS controller
            physics_rules: PhysicsRules
            config: System configuration
        """
        self.llm = llm_interface
        self.rag = rag_database
        self.ris_controller = ris_controller
        self.physics_rules = physics_rules
        self.config = config
        
        # DRL agent (will be set later)
        self.drl_agent: Optional[SACAgent] = None
    
    def strategize(
        self,
        user_A_distance: float,
        user_B_distance: float,
        user_A_angle: float,
        user_B_angle: float,
        user_A_height: float = 0.0,
        user_B_height: float = 0.0,
        current_rates: Optional[Tuple[float, float]] = None
    ) -> Dict:
        """
        Use LLM to analyze scenario and provide strategy.
        
        Args:
            user_A_distance: Distance of user A
            user_B_distance: Distance of user B
            user_A_angle: Angle of user A in degrees
            user_B_angle: Angle of user B in degrees
            user_A_height: Height of user A
            user_B_height: Height of user B
            current_rates: Current achieved rates (if any)
            
        Returns:
            strategy: Dictionary with keys:
                - aperture_size: Number of active RIS elements
                - reward_weights: (α, β, γ) tuple
                - strategy_type: String identifier
                - reasoning: LLM's explanation
        """
        # Create scenario description
        scenario = PhysicsRules.analyze_scenario(
            user_A_distance, user_B_distance,
            user_A_angle, user_B_angle,
            self.config.d_FF
        )
        
        # Retrieve relevant physics rules
        retrieved_rules = self.rag.retrieve(scenario, top_k=3)
        
        # Format rules for LLM
        rules_text = "\n\n".join([
            f"Rule {i+1}: {rule['title']}\n"
            f"Condition: {rule['condition']}\n"
            f"Action: {rule['action']}\n"
            f"Priority: {rule['priority']}"
            for i, (rule, score) in enumerate(retrieved_rules)
        ])
        
        # Create LLM prompt
        system_prompt = """You are an expert in electromagnetic (EM) physics and wireless communications, 
specializing in Reconfigurable Intelligent Surfaces (RIS). Your task is to analyze user scenarios 
and recommend optimal RIS configuration strategies."""
        
        user_prompt = f"""
Given the following scenario:

{scenario}

Height separation: |Δz| = {abs(user_A_height - user_B_height):.1f}m

{"Current rates: Rate_A = " + f"{current_rates[0]:.2f}" + " bps/Hz, Rate_B = " + f"{current_rates[1]:.2f}" + " bps/Hz" if current_rates else ""}

Retrieved Physics Rules:
{rules_text}

Please provide:
1. Recommended RIS aperture size (choose from: 256, 1024, 2048, 3072, 4096)
2. Reward function weights (α, β, γ) for DRL optimization where:
   - α: weight for User A's rate
   - β: weight for User B's rate
   - γ: weight for interference penalty
3. Strategy type identifier
4. Brief reasoning (2-3 sentences)

Respond in JSON format:
{{
    "aperture_size": <integer>,
    "reward_weights": {{"alpha": <float>, "beta": <float>, "gamma": <float>}},
    "strategy_type": "<string>",
    "reasoning": "<string>"
}}
"""
        
        # Query LLM
        response = self.llm.query(user_prompt, system_prompt, temperature=0.3)
        
        # Extract JSON
        strategy = self.llm.extract_json(response)
        
        if strategy is None:
            # Fallback to default
            print("Warning: LLM response invalid, using default strategy")
            strategy = {
                "aperture_size": 2048,
                "reward_weights": {"alpha": 1.0, "beta": 1.0, "gamma": 0.3},
                "strategy_type": "default",
                "reasoning": "Default fallback strategy"
            }
        
        return strategy
    
    def fine_tune_drl(
        self,
        drl_agent: SACAgent,
        environment,
        n_steps: int = 100,
        aperture_mask: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Fine-tune DRL agent with LLM-provided strategy.
        
        Args:
            drl_agent: SAC agent
            environment: Gym environment
            n_steps: Number of fine-tuning steps
            aperture_mask: Active element mask
            
        Returns:
            metrics: Training metrics
        """
        total_reward = 0
        losses = []
        
        for step in range(n_steps):
            # Reset environment
            state, _ = environment.reset()
            
            # Select action
            action = drl_agent.select_action(state, deterministic=False)
            
            # Step
            next_state, reward, done, truncated, info = environment.step(action)
            
            # Store transition
            drl_agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Update
            if len(drl_agent.replay_buffer) > 64:
                loss_dict = drl_agent.update(batch_size=64)
                losses.append(loss_dict)
            
            total_reward += reward
        
        return {
            'total_reward': total_reward,
            'avg_reward': total_reward / n_steps,
            'final_loss': losses[-1] if losses else {}
        }