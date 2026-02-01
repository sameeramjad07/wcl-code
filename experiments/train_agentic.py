"""
Train Agentic-RIS System
=========================

Main training script for LLM-guided DRL optimization.
"""

import sys
sys.path.append('.')

import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

from src.core.system_model import SystemConfig, SystemSimulator
from src.environment.ris_env import RISEnvironment
from src.agents.sac_agent import SACAgent
from src.agents.ppo_agent import PPOAgent
from src.agentic.llm_interface import LLMInterface
from src.agentic.physics_rules import PhysicsRules
from src.agentic.rag_database import RAGDatabase
from src.agentic.agentic_controller import AgenticController
from src.utils.config import load_config


def main():
    # Load configuration
    config_dict = load_config('configs/agentic_config.yaml')
    
    # Create system configuration
    sys_config = SystemConfig(**config_dict['system'])
    print(sys_config.get_info())
    
    # Initialize simulator
    simulator = SystemSimulator(sys_config)
    
    # Initialize LLM interface
    print("\n=== Initializing LLM Interface ===")
    llm = LLMInterface(
        provider=config_dict['llm']['provider'],
        model=config_dict['llm']['model']
    )
    
    # Initialize physics rules
    print("\n=== Loading Physics Rules ===")
    physics_rules = PhysicsRules()

    # Initialize RAG database
    print("=== Initializing RAG Database ===")
    rag = RAGDatabase()
    
    # Load physics rules
    rules_path = Path('data/rag_knowledge/physics_rules.json')
    if rules_path.exists():
        rag.add_rules_from_json(str(rules_path))
        print(f"Loaded {len(rag.rules)} physics rules")
    else:
        print("Warning: Physics rules file not found, using minimal rules")
        from src.agentic.physics_rules import get_default_physics_rules
        for rule in get_default_physics_rules():
            rag.add_rule(rule)
    
    # Initialize agentic controller
    controller = AgenticController(llm, rag, simulator.ris_controller, physics_rules, sys_config)
    
    # Training loop
    print("\n=== Starting Agentic Training ===")
    n_episodes = config_dict['training']['n_episodes']
    llm_update_freq = config_dict['training']['llm_update_frequency']
    fine_tune_steps = config_dict['training']['fine_tune_steps']
    
    # Storage for results
    all_rewards = []
    all_sum_rates = []
    strategies_used = []
    
    # Test distances
    test_distances = config_dict['evaluation']['distances']
    
    for episode in tqdm(range(n_episodes), desc="Training"):
        # Select random distance for this episode
        r_A = np.random.choice(test_distances)
        simulator.set_snapshot_A(r_A)
        
        # Every llm_update_freq episodes, query LLM for strategy
        if episode % llm_update_freq == 0:
            print(f"\n[Episode {episode}] Querying LLM for strategy...")
            
            strategy = controller.strategize(
                user_A_distance=simulator.user_A.r,
                user_B_distance=simulator.user_B.r,
                user_A_angle=np.rad2deg(simulator.user_A.theta),
                user_B_angle=np.rad2deg(simulator.user_B.theta)
            )
            
            print(f"Strategy: {strategy['strategy_type']}")
            print(f"Aperture: {strategy['aperture_size']}")
            print(f"Weights: α={strategy['reward_weights']['alpha']:.2f}, "
                  f"β={strategy['reward_weights']['beta']:.2f}, "
                  f"γ={strategy['reward_weights']['gamma']:.2f}")
            print(f"Reasoning: {strategy['reasoning']}")
            
            # Generate aperture mask
            if strategy['aperture_size']:
                aperture_mask = simulator.ris_controller.generate_aperture_mask(
                    strategy['aperture_size']
                )
            else:
                aperture_mask = None
            
            # Create new environment with LLM strategy
            env = RISEnvironment(
                config=sys_config,
                snapshot_type='A',
                aperture_mask=aperture_mask,
                reward_weights=(
                    strategy['reward_weights']['alpha'],
                    strategy['reward_weights']['beta'],
                    strategy['reward_weights']['gamma']
                )
            )
            
            # Create/update DRL agent
            if episode == 0:
                state_dim = env.observation_space.shape[0]
                action_dim = env.action_space.shape[0]
                
                if config_dict['drl']['agent_type'] == 'sac':
                    agent = SACAgent(state_dim, action_dim, 
                                   hidden_dim=config_dict['drl']['hidden_dim'],
                                   lr=config_dict['drl']['lr'],
                                   gamma=config_dict['drl']['gamma'])
                else:
                    agent = PPOAgent(state_dim, action_dim,
                                   hidden_dim=config_dict['drl']['hidden_dim'],
                                   lr=config_dict['drl']['lr'],
                                   gamma=config_dict['drl']['gamma'])
            
            strategies_used.append(strategy)
        
        # Fine-tune DRL agent
        state, _ = env.reset(options={'r_A': r_A})
        episode_reward = 0
        
        for step in range(fine_tune_steps):
            if config_dict['drl']['agent_type'] == 'sac':
                action = agent.select_action(state)
            else:
                action, log_prob, value = agent.select_action(state)
            
            next_state, reward, done, truncated, info = env.step(action)
            
            if config_dict['drl']['agent_type'] == 'sac':
                agent.replay_buffer.push(state, action, reward, next_state, done)
                if len(agent.replay_buffer) > config_dict['training']['batch_size']:
                    agent.update(config_dict['training']['batch_size'])
            else:
                agent.store_transition(state, action, reward, value, log_prob, done)
            
            episode_reward += reward
            state = next_state
            
            if done or truncated:
                if config_dict['drl']['agent_type'] == 'ppo':
                    agent.update(next_state)
                break
        
        all_rewards.append(episode_reward)
        all_sum_rates.append(info['sum_rate'])
        
        # Logging
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(all_rewards[-50:])
            avg_sum_rate = np.mean(all_sum_rates[-50:])
            print(f"\nEpisode {episode+1}/{n_episodes}")
            print(f"  Avg Reward (last 50): {avg_reward:.3f}")
            print(f"  Avg Sum-Rate (last 50): {avg_sum_rate:.3f} bps/Hz")
    
    # Save results
    results_dir = Path('data/trained_models/agentic')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    agent.save(str(results_dir / 'agent.pt'))
    np.save(results_dir / 'rewards.npy', all_rewards)
    np.save(results_dir / 'sum_rates.npy', all_sum_rates)
    
    with open(results_dir / 'strategies.json', 'w') as f:
        json.dump(strategies_used, f, indent=2)
    
    print(f"\n✓ Training complete! Results saved to {results_dir}")
    print(f"  Final average reward: {np.mean(all_rewards[-100:]):.3f}")
    print(f"  Final average sum-rate: {np.mean(all_sum_rates[-100:]):.3f} bps/Hz")


if __name__ == "__main__":
    main()