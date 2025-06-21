# JanusAI/experiments/emergent_communication_experiment.py v2

"""
Enhanced Emergent Communication Experiment with All Improvements
================================================================

This experiment incorporates:
1. Strong communication cost pressure
2. Compositional structure learning
3. Hard discreteness enforcement
4. Attention-based peer message aggregation
5. Adversarial validation
6. Comprehensive analysis and visualization

Author: JanusAI Team
Date: 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
import logging
import wandb
from tqdm import tqdm
import json
from pathlib import Path

# Import improved components
from integration.improved_learned_comm import (
    ImprovedCommunicationConfig,
    ImprovedSymbolicEncoder,
    AttentionAggregator,
    AdversarialValidator,
    ImprovedCommunicationReward,
    LanguageEvolutionTracker,
    CompositionalEmbeddings,
    plot_communication_analysis
)
from JanusAI.integration.schemas import AgentRole

logger = logging.getLogger(__name__)


@dataclass
class EnhancedExperimentConfig:
    """Configuration for enhanced experiment."""
    # Basic settings
    experiment_name: str = "enhanced_emergent_comm"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Agent configuration
    num_agents: int = 8
    num_explorers: int = 3
    num_refiners: int = 3
    num_validators: int = 2
    
    # Communication settings
    vocab_size: int = 64
    max_message_length: int = 6
    symbol_cost: float = 0.02
    length_penalty: float = 0.05
    
    # Training configuration
    num_episodes: int = 1500
    episode_length: int = 100
    batch_size: int = 16
    learning_rate: float = 5e-4
    gradient_clip: float = 1.0
    
    # Curriculum phases
    phase1_episodes: int = 300  # No communication
    phase2_episodes: int = 500  # Tactical only
    phase3_episodes: int = 700  # Full communication
    
    # Loss weights
    task_weight: float = 0.4
    comm_weight: float = 0.3
    structure_weight: float = 0.2
    adversarial_weight: float = 0.1
    
    # Analysis settings
    log_interval: int = 20
    save_interval: int = 100
    analysis_interval: int = 200
    
    # Paths
    output_dir: str = "experiments/enhanced_comm"
    checkpoint_dir: str = "checkpoints"
    
    # Logging
    use_wandb: bool = True
    wandb_project: str = "janus-enhanced-comm"
    verbose: bool = True


class EnhancedCommunicationAgent(nn.Module):
    """
    Agent with all communication improvements integrated.
    """
    
    def __init__(self,
                 agent_id: str,
                 role: AgentRole,
                 obs_dim: int,
                 action_dim: int,
                 encoder: ImprovedSymbolicEncoder,
                 aggregator: AttentionAggregator,
                 config: ImprovedCommunicationConfig):
        super().__init__()
        
        self.agent_id = agent_id
        self.role = role
        self.encoder = encoder  # Shared encoder
        self.aggregator = aggregator  # Shared aggregator
        self.config = config
        
        # Policy network (incorporates aggregated communications)
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim + config.latent_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim + config.latent_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Role-specific exploration
        self.exploration_schedule = self._create_exploration_schedule()
        
        # Agent memory
        self.episode_buffer = []
        
    def _create_exploration_schedule(self) -> callable:
        """Create role-specific exploration schedule."""
        base_rates = {
            AgentRole.EXPLORER: 0.4,
            AgentRole.REFINER: 0.15,
            AgentRole.VALIDATOR: 0.05,
            AgentRole.SPECIALIST: 0.25
        }
        base_rate = base_rates.get(self.role, 0.2)
        
        def schedule(episode: int) -> float:
            # Decay exploration over time
            decay_factor = 0.995 ** episode
            min_rate = base_rate * 0.1
            return max(base_rate * decay_factor, min_rate)
        
        return schedule
    
    def encode_observation(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode observation to communication."""
        return self.encoder(obs)
    
    def aggregate_communications(self, 
                               own_state: torch.Tensor,
                               peer_messages: List[torch.Tensor]) -> torch.Tensor:
        """Aggregate peer communications using attention."""
        if not peer_messages:
            return torch.zeros(self.config.latent_dim, device=own_state.device)
        
        aggregated, weights = self.aggregator(own_state, peer_messages)
        return aggregated
    
    def act(self, 
            obs: torch.Tensor,
            peer_messages: List[torch.Tensor],
            episode: int) -> Dict[str, Any]:
        """Select action with communication."""
        # Get exploration rate
        exploration_rate = self.exploration_schedule(episode)
        
        # Aggregate peer communications
        latent_obs = obs[:self.config.latent_dim]  # Use part of obs as latent
        aggregated_comm = self.aggregate_communications(latent_obs, peer_messages)
        
        # Combine observation and communication
        combined_input = torch.cat([obs, aggregated_comm])
        
        # Get policy and value
        with torch.no_grad():
            action_logits = self.policy_net(combined_input)
            action_probs = F.softmax(action_logits, dim=-1)
            value = self.value_net(combined_input)
        
        # Sample action
        if torch.rand(1).item() < exploration_rate:
            action = torch.randint(0, action_probs.shape[-1], (1,)).item()
        else:
            action = torch.multinomial(action_probs, 1).item()
        
        return {
            'action': action,
            'action_probs': action_probs,
            'value': value,
            'aggregated_comm': aggregated_comm
        }
    
    def store_transition(self, transition: Dict[str, Any]):
        """Store transition in episode buffer."""
        self.episode_buffer.append(transition)
    
    def clear_buffer(self):
        """Clear episode buffer."""
        self.episode_buffer = []


class EnhancedEmergentCommExperiment:
    """
    Main experiment class with all improvements integrated.
    """
    
    def __init__(self, config: EnhancedExperimentConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Set random seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Create output directories
        self.setup_directories()
        
        # Initialize communication components
        self.setup_communication_system()
        
        # Initialize agents
        self.agents = self.create_agents()
        
        # Initialize adversarial validator
        self.adversarial_validator = AdversarialValidator(
            message_dim=config.vocab_size * config.max_message_length,
            hidden_dim=128
        ).to(self.device)
        
        # Initialize tracking
        self.language_tracker = LanguageEvolutionTracker(
            vocab_size=config.vocab_size,
            max_phases=3
        )
        
        # Initialize optimizers
        self.setup_optimizers()
        
        # Training state
        self.episode = 0
        self.global_step = 0
        self.current_phase = 1
        
        # Initialize logging
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                name=config.experiment_name,
                config=config.__dict__
            )
    
    def setup_directories(self):
        """Create experiment directories."""
        base_path = Path(self.config.output_dir)
        base_path.mkdir(parents=True, exist_ok=True)
        
        (base_path / self.config.checkpoint_dir).mkdir(exist_ok=True)
        (base_path / "analysis").mkdir(exist_ok=True)
        (base_path / "logs").mkdir(exist_ok=True)
    
    def setup_communication_system(self):
        """Initialize communication components."""
        comm_config = ImprovedCommunicationConfig(
            vocab_size=self.config.vocab_size,
            max_message_length=self.config.max_message_length,
            symbol_cost=self.config.symbol_cost,
            length_penalty=self.config.length_penalty,
            use_compositional_prior=True,
            use_attention_aggregation=True,
            use_adversarial_validator=True
        )
        
        # Shared encoder (all agents use same encoding)
        self.encoder = ImprovedSymbolicEncoder(
            input_dim=self.get_obs_dim(),
            config=comm_config
        ).to(self.device)
        
        # Shared aggregator
        self.aggregator = AttentionAggregator(comm_config).to(self.device)
        
        # Compositional embeddings
        self.compositional_embeds = CompositionalEmbeddings(
            vocab_size=self.config.vocab_size,
            embed_dim=16
        ).to(self.device)
        
        # Reward function
        self.reward_fn = ImprovedCommunicationReward(comm_config)
        
        self.comm_config = comm_config
    
    def get_obs_dim(self) -> int:
        """Get observation dimension."""
        # Simplified for example
        return 64
    
    def get_action_dim(self) -> int:
        """Get action dimension."""
        # Simplified for example
        return 10
    
    def create_agents(self) -> List[EnhancedCommunicationAgent]:
        """Create agents with roles."""
        agents = []
        
        # Create agents by role
        role_counts = [
            (AgentRole.EXPLORER, self.config.num_explorers),
            (AgentRole.REFINER, self.config.num_refiners),
            (AgentRole.VALIDATOR, self.config.num_validators)
        ]
        
        for role, count in role_counts:
            for i in range(count):
                agent = EnhancedCommunicationAgent(
                    agent_id=f"{role.value}_{i:03d}",
                    role=role,
                    obs_dim=self.get_obs_dim(),
                    action_dim=self.get_action_dim(),
                    encoder=self.encoder,
                    aggregator=self.aggregator,
                    config=self.comm_config
                ).to(self.device)
                agents.append(agent)
        
        return agents
    
    def setup_optimizers(self):
        """Setup optimizers for all components."""
        # Collect parameters
        params = []
        
        # Communication system
        params.extend(self.encoder.parameters())
        params.extend(self.aggregator.parameters())
        params.extend(self.compositional_embeds.parameters())
        
        # Adversarial validator
        params.extend(self.adversarial_validator.parameters())
        
        # Agent networks
        for agent in self.agents:
            params.extend(agent.policy_net.parameters())
            params.extend(agent.value_net.parameters())
        
        # Main optimizer
        self.optimizer = optim.Adam(params, lr=self.config.learning_rate)
        
        # Adversarial optimizer (separate for stability)
        self.adv_optimizer = optim.Adam(
            self.adversarial_validator.parameters(),
            lr=self.config.learning_rate * 0.5
        )
    
    def get_current_phase(self) -> int:
        """Determine current training phase."""
        if self.episode < self.config.phase1_episodes:
            return 1
        elif self.episode < self.config.phase1_episodes + self.config.phase2_episodes:
            return 2
        else:
            return 3
    
    def run_episode(self) -> Dict[str, float]:
        """Run one episode with all improvements."""
        self.current_phase = self.get_current_phase()
        enable_comm = self.current_phase >= 2
        enable_strategic = self.current_phase >= 3
        
        # Episode setup
        observations = [torch.randn(self.get_obs_dim()).to(self.device) 
                       for _ in self.agents]
        
        # Episode metrics
        episode_metrics = {
            'rewards': [],
            'comm_losses': [],
            'structure_losses': [],
            'adversarial_scores': [],
            'message_lengths': [],
            'task_successes': 0,
            'discoveries': []
        }
        
        # Run episode steps
        for step in range(self.config.episode_length):
            # Phase 1: Encode observations (if communication enabled)
            messages = []
            encode_results = []
            
            if enable_comm:
                for i, (agent, obs) in enumerate(zip(self.agents, observations)):
                    encode_result = agent.encode_observation(obs)
                    encode_results.append(encode_result)
                    messages.append(encode_result['message'])
                    
                    # Track language evolution
                    if step % 10 == 0:  # Sample periodically
                        self.language_tracker.record_communication(
                            phase=self.current_phase,
                            message=encode_result['message'],
                            expression=f"state_{i}_{step}",
                            task_success=False,  # Will update later
                            episode=self.episode
                        )
            
            # Phase 2: Agents act with aggregated communications
            actions = []
            transitions = []
            
            for i, agent in enumerate(self.agents):
                # Get peer messages (excluding own)
                peer_messages = []
                if enable_comm:
                    for j, msg in enumerate(messages):
                        if i != j:
                            # Extract latent representation from message
                            peer_latent = self.compositional_embeds(msg.argmax(dim=-1))
                            peer_latent = peer_latent.mean(dim=1)  # Average over sequence
                            peer_messages.append(peer_latent)
                
                # Act
                action_result = agent.act(
                    observations[i],
                    peer_messages,
                    self.episode
                )
                actions.append(action_result['action'])
                
                # Store transition
                transition = {
                    'obs': observations[i],
                    'action': action_result['action'],
                    'action_probs': action_result['action_probs'],
                    'value': action_result['value'],
                    'message': messages[i] if enable_comm else None
                }
                transitions.append(transition)
                agent.store_transition(transition)
            
            # Phase 3: Environment step (simplified)
            rewards = []
            next_observations = []
            
            for i in range(len(self.agents)):
                # Simulate reward based on action and role
                base_reward = torch.rand(1).item() * 0.5
                
                # Role-specific bonus
                if self.agents[i].role == AgentRole.EXPLORER:
                    exploration_bonus = 0.1 if actions[i] > 5 else 0
                    reward = base_reward + exploration_bonus
                elif self.agents[i].role == AgentRole.REFINER:
                    refinement_bonus = 0.2 if actions[i] < 3 else 0
                    reward = base_reward + refinement_bonus
                else:  # Validator
                    validation_bonus = 0.15 if actions[i] == 0 else 0
                    reward = base_reward + validation_bonus
                
                rewards.append(reward)
                
                # Check for discovery
                if reward > 0.7:
                    episode_metrics['task_successes'] += 1
                    episode_metrics['discoveries'].append({
                        'agent': self.agents[i].agent_id,
                        'step': step,
                        'reward': reward
                    })
                
                # Next observation
                next_obs = torch.randn(self.get_obs_dim()).to(self.device)
                next_observations.append(next_obs)
            
            # Phase 4: Compute losses (if communication enabled)
            if enable_comm and encode_results:
                # Communication reconstruction loss
                comm_losses = []
                for i, encode_result in enumerate(encode_results):
                    # Decode own message
                    message_tokens = encode_result['message'].argmax(dim=-1)
                    embedded = self.compositional_embeds(message_tokens)
                    
                    # Simple reconstruction (in practice, use proper decoder)
                    reconstructed = embedded.mean(dim=1).repeat(1, self.get_obs_dim() // embedded.size(-1) + 1)
                    reconstructed = reconstructed[:, :self.get_obs_dim()]
                    
                    reconstruction_loss = F.mse_loss(reconstructed, observations[i])
                    comm_losses.append(reconstruction_loss)
                
                # Compositional structure loss
                structure_losses = []
                for encode_result in encode_results:
                    message_tokens = encode_result['message'].argmax(dim=-1)
                    struct_loss = self.compositional_embeds.get_compositional_loss(message_tokens)
                    structure_losses.append(struct_loss)
                
                # Adversarial validation
                if enable_strategic:
                    real_messages = torch.stack(messages)
                    adv_loss = self.adversarial_validator.compute_adversarial_loss(real_messages)
                    
                    # Get adversarial scores
                    with torch.no_grad():
                        validity_scores = self.adversarial_validator(real_messages)
                        validity_probs = F.softmax(validity_scores, dim=-1)
                        adversarial_scores = validity_probs[:, 0]  # Probability of being valid
                        episode_metrics['adversarial_scores'].extend(adversarial_scores.cpu().numpy())
                
                # Communication cost
                for encode_result in encode_results:
                    actual_length = encode_result['actual_lengths'].mean()
                    episode_metrics['message_lengths'].append(actual_length.item())
                
                # Store losses
                if comm_losses:
                    episode_metrics['comm_losses'].append(torch.stack(comm_losses).mean().item())
                if structure_losses:
                    episode_metrics['structure_losses'].append(torch.stack(structure_losses).mean().item())
            
            # Update observations
            observations = next_observations
            episode_metrics['rewards'].extend(rewards)
        
        # Compute episode summary
        summary = {
            'episode': self.episode,
            'phase': self.current_phase,
            'avg_reward': np.mean(episode_metrics['rewards']),
            'num_discoveries': len(episode_metrics['discoveries']),
            'task_success_rate': episode_metrics['task_successes'] / (self.config.episode_length * len(self.agents))
        }
        
        if enable_comm:
            summary.update({
                'avg_comm_loss': np.mean(episode_metrics['comm_losses']) if episode_metrics['comm_losses'] else 0,
                'avg_structure_loss': np.mean(episode_metrics['structure_losses']) if episode_metrics['structure_losses'] else 0,
                'avg_message_length': np.mean(episode_metrics['message_lengths']) if episode_metrics['message_lengths'] else 0
            })
            
            if enable_strategic:
                summary['avg_adversarial_score'] = np.mean(episode_metrics['adversarial_scores']) if episode_metrics['adversarial_scores'] else 0
        
        # Get language statistics
        if enable_comm:
            phase_summary = self.language_tracker.get_phase_summary(self.current_phase)
            summary.update({
                f'phase{self.current_phase}_symbol_entropy': phase_summary['symbol_entropy'],
                f'phase{self.current_phase}_unique_symbols': phase_summary['unique_symbols_used'],
                f'phase{self.current_phase}_consistency': phase_summary['expression_consistency']
            })
        
        return summary
    
    def train_step(self, episode_data: Dict[str, Any]):
        """Perform training update."""
        # Compute gradients and update
        self.optimizer.zero_grad()
        
        # Placeholder for actual loss computation
        # In practice, compute proper policy gradient or actor-critic loss
        
        # Update temperature
        if self.current_phase >= 2:
            self.encoder.update_temperature()
    
    def run_analysis(self):
        """Run comprehensive analysis of communication evolution."""
        logger.info("Running communication analysis...")
        
        # Create visualizations
        output_dir = Path(self.config.output_dir) / "analysis"
        plot_communication_analysis(
            self.language_tracker,
            save_prefix=str(output_dir / f"analysis_ep{self.episode}")
        )
        
        # Save emergent dictionary
        dictionary = self.language_tracker.get_emergent_dictionary()
        with open(output_dir / f"dictionary_ep{self.episode}.json", 'w') as f:
            json.dump(dictionary, f, indent=2)
        
        # Log to wandb if enabled
        if self.config.use_wandb:
            wandb.log({
                "emergent_patterns": len(dictionary),
                "phase_summaries": {
                    f"phase_{i}": self.language_tracker.get_phase_summary(i)
                    for i in range(1, 4)
                }
            })
    
    def save_checkpoint(self, tag: str = "latest"):
        """Save experiment checkpoint."""
        checkpoint_path = Path(self.config.output_dir) / self.config.checkpoint_dir / f"checkpoint_{tag}.pt"
        
        checkpoint = {
            'episode': self.episode,
            'phase': self.current_phase,
            'encoder_state': self.encoder.state_dict(),
            'aggregator_state': self.aggregator.state_dict(),
            'adversarial_state': self.adversarial_validator.state_dict(),
            'agent_states': [agent.state_dict() for agent in self.agents],
            'optimizer_state': self.optimizer.state_dict(),
            'language_tracker': self.language_tracker,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, tag: str = "latest"):
        """Load experiment checkpoint."""
        checkpoint_path = Path(self.config.output_dir) / self.config.checkpoint_dir / f"checkpoint_{tag}.pt"
        
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint {checkpoint_path} not found")
            return False
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.episode = checkpoint['episode']
        self.current_phase = checkpoint['phase']
        self.encoder.load_state_dict(checkpoint['encoder_state'])
        self.aggregator.load_state_dict(checkpoint['aggregator_state'])
        self.adversarial_validator.load_state_dict(checkpoint['adversarial_state'])
        
        for agent, state in zip(self.agents, checkpoint['agent_states']):
            agent.load_state_dict(state)
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.language_tracker = checkpoint['language_tracker']
        
        logger.info(f"Loaded checkpoint from episode {self.episode}")
        return True
    
    def train(self):
        """Main training loop with all improvements."""
        logger.info(f"Starting enhanced emergent communication experiment")
        logger.info(f"Configuration: {self.config}")
        
        best_performance = 0
        episode_summaries = []
        
        for episode in tqdm(range(self.config.num_episodes), desc="Training"):
            self.episode = episode
            
            # Run episode
            summary = self.run_episode()
            episode_summaries.append(summary)
            
            # Training step
            self.train_step(summary)
            
            # Logging
            if episode % self.config.log_interval == 0:
                logger.info(f"Episode {episode}: {summary}")
                
                if self.config.use_wandb:
                    wandb.log(summary)
            
            # Analysis
            if episode % self.config.analysis_interval == 0 and episode > 0:
                self.run_analysis()
            
            # Checkpointing
            if episode % self.config.save_interval == 0:
                self.save_checkpoint(tag=f"ep{episode}")
                
                # Save best model
                current_performance = summary['avg_reward'] * summary.get('task_success_rate', 1.0)
                if current_performance > best_performance:
                    best_performance = current_performance
                    self.save_checkpoint(tag="best")
        
        # Final analysis
        logger.info("Running final analysis...")
        self.run_analysis()
        
        # Save final results
        results = {
            'config': self.config.__dict__,
            'episode_summaries': episode_summaries,
            'final_language_stats': {
                f"phase_{i}": self.language_tracker.get_phase_summary(i)
                for i in range(1, 4)
            },
            'emergent_dictionary': self.language_tracker.get_emergent_dictionary()
        }
        
        with open(Path(self.config.output_dir) / "final_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("Experiment complete!")
        
        return results


def main():
    """Run enhanced experiment."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Emergent Communication Experiment")
    parser.add_argument("--num_agents", type=int, default=8, help="Number of agents")
    parser.add_argument("--vocab_size", type=int, default=64, help="Vocabulary size")
    parser.add_argument("--num_episodes", type=int, default=1500, help="Number of episodes")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--experiment_name", type=str, default="enhanced_comm", help="Experiment name")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Load from checkpoint")
    
    args = parser.parse_args()
    
    # Create configuration
    config = EnhancedExperimentConfig(
        experiment_name=args.experiment_name,
        num_agents=args.num_agents,
        vocab_size=args.vocab_size,
        num_episodes=args.num_episodes,
        use_wandb=args.use_wandb
    )
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{config.output_dir}/experiment.log"),
            logging.StreamHandler()
        ]
    )
    
    # Create and run experiment
    experiment = EnhancedEmergentCommExperiment(config)
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        experiment.load_checkpoint(args.load_checkpoint)
    
    # Run training
    results = experiment.train()
    
    # Print final summary
    print("\n" + "="*50)
    print("EXPERIMENT COMPLETE")
    print("="*50)
    print(f"Final Performance: {results['episode_summaries'][-1]['avg_reward']:.3f}")
    print(f"Unique Symbols Used: {results['final_language_stats']['phase_3']['unique_symbols_used']}")
    print(f"Symbol Entropy: {results['final_language_stats']['phase_3']['symbol_entropy']:.3f}")
    print(f"Expression Consistency: {results['final_language_stats']['phase_3']['expression_consistency']:.3f}")
    print(f"Emergent Patterns: {len(results['emergent_dictionary'])}")
    print("="*50)


if __name__ == "__main__":
    main()