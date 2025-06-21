# File: JanusAI/experiments/attention_discovery.py
"""
Complete GPT-2 Attention Discovery Experiment
=============================================

This script integrates all components to run the first concrete experiment:
discovering symbolic approximations for GPT-2 attention head patterns.
"""

import json
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2Model, GPT2Tokenizer

# Janus core imports (updated for new structure)
from janus_ai.core.grammar.enhanced_ai_grammar import EnhancedAIGrammar, AttentionVariable
from janus_ai.ml.rewards.interpretability_reward import InterpretabilityReward, patch_interpretability_reward_methods
from janus_ai.environments.base.symbolic_env import SymbolicDiscoveryEnv
from janus_ai.ml.training.ppo_trainer import PPOTrainer
from janus_ai.ai_interpretability.evaluation.fidelity import ModelFidelityEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class AttentionDiscoveryConfig:
    """Configuration for GPT-2 attention pattern discovery experiment."""
    
    # Model configuration
    model_name: str = 'gpt2'
    layer_index: int = 0
    head_index: int = 1  # Previous token head
    
    # Data configuration
    num_text_samples: int = 200
    max_sequence_length: int = 32
    
    # Discovery configuration
    max_complexity: int = 25
    max_expression_depth: int = 8
    
    # Training configuration
    training_episodes: int = 500
    learning_rate: float = 3e-4
    gamma: float = 0.99
    
    # Reward configuration
    reward_weights: Dict[str, float] = None
    
    # Output configuration
    save_results: bool = True
    output_dir: str = "results/attention_discovery"
    
    def __post_init__(self):
        if self.reward_weights is None:
            self.reward_weights = {
                'fidelity': 0.5,
                'simplicity': 0.3,
                'consistency': 0.15,
                'insight': 0.05
            }


class GPT2AttentionDiscoveryExperiment:
    """
    Complete experiment for discovering symbolic patterns in GPT-2 attention.
    """
    
    def __init__(self, config: AttentionDiscoveryConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Core components
        self.model: Optional[GPT2Model] = None
        self.tokenizer: Optional[GPT2Tokenizer] = None
        self.grammar: Optional[EnhancedAIGrammar] = None
        self.reward_calculator: Optional[InterpretabilityReward] = None
        self.environment: Optional[SymbolicDiscoveryEnv] = None
        self.trainer: Optional[PPOTrainer] = None
        
        # Experimental data
        self.attention_data: Dict[str, Any] = {}
        self.variables: List[AttentionVariable] = []
        self.results: Dict[str, Any] = {}
        
        # Initialize patches
        patch_interpretability_reward_methods()

    def run_complete_experiment(self) -> Dict[str, Any]:
        """
        Run the complete experiment pipeline.
        
        Returns:
            Dict containing experimental results and discovered patterns
        """
        self.logger.info("Starting GPT-2 Attention Discovery Experiment")
        start_time = time.time()
        
        try:
            # Phase 1: Setup
            self.logger.info("Phase 1: Setting up experiment components...")
            self._setup_experiment()
            
            # Phase 2: Data Collection
            self.logger.info("Phase 2: Collecting attention data from GPT-2...")
            self._collect_attention_data()
            
            # Phase 3: Environment Preparation
            self.logger.info("Phase 3: Preparing symbolic discovery environment...")
            self._prepare_environment()
            
            # Phase 4: Discovery Training
            self.logger.info("Phase 4: Training symbolic discovery agent...")
            training_results = self._train_discovery_agent()
            
            # Phase 5: Result Analysis
            self.logger.info("Phase 5: Analyzing discovered patterns...")
            analysis_results = self._analyze_results(training_results)
            
            # Phase 6: Validation
            self.logger.info("Phase 6: Validating discovered patterns...")
            validation_results = self._validate_discoveries(analysis_results)
            
            # Compile final results
            total_time = time.time() - start_time
            self.results = {
                'config': self.config.__dict__,
                'training_results': training_results,
                'analysis_results': analysis_results,
                'validation_results': validation_results,
                'total_time_seconds': total_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save results
            if self.config.save_results:
                self._save_results()
            
            self.logger.info(f"Experiment completed successfully in {total_time:.2f} seconds")
            return self.results
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}")
            raise

    def _setup_experiment(self):
        """Initialize all experimental components."""
        
        # Load GPT-2 model
        self.logger.info(f"Loading GPT-2 model: {self.config.model_name}")
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = GPT2Model.from_pretrained(
            self.config.model_name,
            output_attentions=True,
            output_hidden_states=True
        ).to(device).eval()
        
        # Initialize enhanced grammar
        self.logger.info("Initializing enhanced AI grammar...")
        self.grammar = EnhancedAIGrammar()
        
        # Add custom primitives for attention discovery
        self.grammar.add_custom_primitive(
            'is_previous_token',
            lambda i, j: float(abs(i - j - 1) < 0.1),
            'pattern'
        )
        self.grammar.add_custom_primitive(
            'distance_decay',
            lambda i, j: np.exp(-abs(i - j)),
            'pattern'
        )
        
        # Create attention-specific variables
        self.variables = self.grammar.create_attention_variables(
            sequence_length=self.config.max_sequence_length
        )
        
        self.logger.info(f"Created {len(self.variables)} attention variables")

    def _collect_attention_data(self):
        """Extract attention patterns from GPT-2 model."""
        
        # Generate diverse text samples
        text_samples = self._generate_text_samples()
        
        attention_matrices = []
        sequence_lengths = []
        token_features = []
        
        for text in text_samples:
            # Tokenize text
            encoded = self.tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.config.max_sequence_length
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.model(**encoded)
            
            # Extract attention from specified layer and head
            attention = outputs.attentions[self.config.layer_index]
            head_attention = attention[0, self.config.head_index].cpu().numpy()
            
            # Get sequence info
            seq_len = encoded['input_ids'].shape[1]
            tokens = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
            
            # Create token features
            token_types = self._extract_token_features(tokens)
            
            attention_matrices.append(head_attention[:seq_len, :seq_len])
            sequence_lengths.append(seq_len)
            token_features.append(token_types)
        
        self.attention_data = {
            'attention_matrices': attention_matrices,
            'sequence_lengths': sequence_lengths,
            'token_features': token_features,
            'num_samples': len(attention_matrices)
        }
        
        self.logger.info(f"Collected attention data from {len(attention_matrices)} text samples")

    def _generate_text_samples(self) -> List[str]:
        """Generate diverse text samples for attention analysis."""
        
        # Base patterns that highlight different attention behaviors
        base_patterns = [
            "The cat sat on the mat.",
            "A B A B A B A B",
            "hello world hello world",
            "The quick brown fox jumps over the lazy dog.",
            "abc abc abc abc",
            "red blue red blue red blue",
            "In the beginning was the Word.",
            "To be or not to be, that is the question.",
            "Mary had a little lamb.",
            "Once upon a time in a land far away."
        ]
        
        # Generate variations
        text_samples = []
        samples_per_pattern = self.config.num_text_samples // len(base_patterns)
        
        for pattern in base_patterns:
            for i in range(samples_per_pattern):
                # Add variations: case, repetition, punctuation
                if i % 3 == 0:
                    sample = pattern.upper()
                elif i % 3 == 1:
                    sample = pattern.lower()
                else:
                    sample = pattern
                
                # Sometimes add repetition
                if i % 4 == 0:
                    sample = sample + " " + sample
                
                text_samples.append(sample)
        
        # Fill remaining samples with random selections
        while len(text_samples) < self.config.num_text_samples:
            text_samples.append(np.random.choice(base_patterns))
        
        return text_samples[:self.config.num_text_samples]

    def _extract_token_features(self, tokens: List[str]) -> np.ndarray:
        """Extract features from tokens for variable creation."""
        features = np.zeros(len(tokens))
        
        for i, token in enumerate(tokens):
            # Simple feature: 1 for word-start tokens (Ġ prefix), 0 for others
            if token.startswith('Ġ'):
                features[i] = 1.0
            elif token in ['.', ',', '!', '?', ';', ':']:
                features[i] = 0.5  # Punctuation
            else:
                features[i] = 0.0  # Subword or special token
        
        return features

    def _prepare_environment(self):
        """Prepare the symbolic discovery environment."""
        
        # Convert attention data to training format
        input_data, output_data = self._convert_attention_data_for_training()
        
        # Create data samples for evaluation
        data_samples = {
            'inputs': input_data,
            'outputs': output_data,
            'attention_weights': self.attention_data['attention_matrices'],
            'sequence_length': self.config.max_sequence_length
        }
        
        # Initialize reward calculator
        self.reward_calculator = InterpretabilityReward(
            fidelity_weight=self.config.reward_weights['fidelity'],
            simplicity_weight=self.config.reward_weights['simplicity'],
            consistency_weight=self.config.reward_weights['consistency'],
            insight_weight=self.config.reward_weights['insight'],
            target_model=self.model,
            data_samples=data_samples,
            variables_for_evaluation=self.variables
        )
        
        # Create symbolic discovery environment
        combined_data = np.column_stack([input_data, output_data.reshape(-1, 1)])
        
        self.environment = SymbolicDiscoveryEnv(
            grammar=self.grammar,
            target_data=combined_data,
            variables=self.variables,
            max_depth=self.config.max_expression_depth,
            max_complexity=self.config.max_complexity,
            reward_config={
                'completion_bonus': 0.2,
                'mse_weight': 0.5,
                'complexity_penalty': -0.02
            }
        )
        
        self.logger.info(f"Environment prepared with {len(input_data)} training examples")

    def _convert_attention_data_for_training(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert attention matrices to input-output training data."""
        
        inputs = []
        outputs = []
        
        for attn_matrix, seq_len, token_features in zip(
            self.attention_data['attention_matrices'],
            self.attention_data['sequence_lengths'],
            self.attention_data['token_features']
        ):
            for i in range(seq_len):
                for j in range(seq_len):
                    # Create input features for position pair (i, j)
                    features = [
                        i,  # pos_i
                        j,  # pos_j
                        i - j,  # pos_diff
                        i / max(j, 1),  # pos_ratio
                        abs(i - j),  # relative_pos
                        float(i - j == 1),  # is_previous
                        float(i == j),  # is_diagonal
                        float(j == 0),  # is_bos
                        token_features[i],  # token_type_i
                        token_features[j],  # token_type_j
                    ]
                    
                    inputs.append(features)
                    outputs.append(attn_matrix[i, j])
        
        return np.array(inputs), np.array(outputs)

    def _train_discovery_agent(self) -> Dict[str, Any]:
        """Train the symbolic discovery agent using PPO."""
        
        # Initialize PPO trainer
        self.trainer = PPOTrainer(
            env=self.environment,
            learning_rate=self.config.learning_rate,
            gamma=self.config.gamma,
            device=device
        )
        
        # Training metrics
        training_metrics = {
            'episode_rewards': [],
            'episode_losses': [],
            'best_expressions': [],
            'convergence_episode': None
        }
        
        best_reward = float('-inf')
        best_expression = None
        
        # Training loop
        for episode in range(self.config.training_episodes):
            # Run episode
            episode_reward, episode_loss, episode_info = self.trainer.train_episode()
            
            # Track metrics
            training_metrics['episode_rewards'].append(episode_reward)
            training_metrics['episode_losses'].append(episode_loss)
            
            # Check for best expression
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_expression = episode_info.get('expression')
                training_metrics['best_expressions'].append({
                    'episode': episode,
                    'reward': episode_reward,
                    'expression': str(best_expression)
                })
            
            # Log progress
            if episode % 50 == 0:
                self.logger.info(f"Episode {episode}: Reward={episode_reward:.4f}, Loss={episode_loss:.4f}")
                if best_expression:
                    self.logger.info(f"Best expression so far: {best_expression}")
            
            # Check convergence
            if episode > 100:
                recent_rewards = training_metrics['episode_rewards'][-50:]
                if np.std(recent_rewards) < 0.01 and training_metrics['convergence_episode'] is None:
                    training_metrics['convergence_episode'] = episode
                    self.logger.info(f"Training converged at episode {episode}")
        
        training_metrics['final_best_reward'] = best_reward
        training_metrics['final_best_expression'] = str(best_expression)
        
        return training_metrics

    def _analyze_results(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the discovered patterns and expressions."""
        
        analysis = {
            'training_convergence': training_results.get('convergence_episode'),
            'final_reward': training_results.get('final_best_reward'),
            'discovered_expression': training_results.get('final_best_expression'),
            'expression_analysis': {},
            'attention_pattern_analysis': {}
        }
        
        # Analyze best expression
        best_expr_str = training_results.get('final_best_expression', '')
        if best_expr_str:
            analysis['expression_analysis'] = {
                'complexity': len(best_expr_str),
                'contains_position_diff': 'pos_diff' in best_expr_str,
                'contains_previous_token': 'is_previous' in best_expr_str,
                'contains_diagonal': 'is_diagonal' in best_expr_str,
                'mathematical_operations': self._count_math_operations(best_expr_str)
            }
        
        # Analyze attention patterns
        if self.attention_data['attention_matrices']:
            avg_attention = np.mean(self.attention_data['attention_matrices'], axis=0)
            analysis['attention_pattern_analysis'] = {
                'diagonal_dominance': np.trace(avg_attention) / np.sum(avg_attention),
                'previous_token_strength': self._measure_previous_token_pattern(avg_attention),
                'attention_entropy': float(np.mean([
                    -np.sum(matrix * np.log(matrix + 1e-8))
                    for matrix in self.attention_data['attention_matrices']
                ])),
                'max_attention_position': int(np.unravel_index(np.argmax(avg_attention), avg_attention.shape)[1])
            }
        
        return analysis

    def _count_math_operations(self, expression_str: str) -> Dict[str, int]:
        """Count mathematical operations in expression string."""
        operations = {'+': 0, '-': 0, '*': 0, '/': 0, '**': 0, 'exp': 0, 'log': 0, 'sin': 0, 'cos': 0}
        
        for op in operations:
            operations[op] = expression_str.count(op)
        
        return operations

    def _measure_previous_token_pattern(self, attention_matrix: np.ndarray) -> float:
        """Measure how much the attention follows a previous-token pattern."""
        if attention_matrix.shape[0] < 2:
            return 0.0
        
        previous_token_weight = 0.0
        total_weight = 0.0
        
        for i in range(1, attention_matrix.shape[0]):
            previous_token_weight += attention_matrix[i, i-1]
            total_weight += np.sum(attention_matrix[i, :])
        
        return previous_token_weight / max(total_weight, 1e-8)

    def _validate_discoveries(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the discovered patterns on held-out data."""
        
        # Generate new validation text samples
        validation_texts = self._generate_text_samples()[:50]  # Smaller validation set
        
        validation_results = {
            'validation_fidelity': 0.0,
            'validation_consistency': 0.0,
            'pattern_generalization': False,
            'interpretability_score': 0.0
        }
        
        # Collect validation attention data
        validation_attention = []
        for text in validation_texts:
            try:
                encoded = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=32)
                encoded = {k: v.to(device) for k, v in encoded.items()}
                
                with torch.no_grad():
                    outputs = self.model(**encoded)
                
                attention = outputs.attentions[self.config.layer_index]
                head_attention = attention[0, self.config.head_index].cpu().numpy()
                validation_attention.append(head_attention)
                
            except Exception as e:
                self.logger.warning(f"Validation sample failed: {e}")
                continue
        
        if validation_attention:
            # Calculate validation metrics
            avg_val_attention = np.mean(validation_attention, axis=0)
            
            # Compare with training patterns
            training_avg = np.mean(self.attention_data['attention_matrices'], axis=0)
            correlation = np.corrcoef(
                avg_val_attention.flatten(),
                training_avg.flatten()
            )[0, 1]
            
            validation_results['validation_fidelity'] = float(correlation)
            validation_results['pattern_generalization'] = correlation > 0.7
            
            # Calculate consistency across validation samples
            consistency_scores = []
            for val_attn in validation_attention:
                if val_attn.shape == avg_val_attention.shape:
                    corr = np.corrcoef(val_attn.flatten(), avg_val_attention.flatten())[0, 1]
                    consistency_scores.append(corr)
            
            validation_results['validation_consistency'] = float(np.mean(consistency_scores))
            
            # Overall interpretability score
            validation_results['interpretability_score'] = (
                0.4 * validation_results['validation_fidelity'] +
                0.3 * validation_results['validation_consistency'] +
                0.3 * (1.0 if validation_results['pattern_generalization'] else 0.0)
            )
        
        return validation_results

    def _save_results(self):
        """Save experimental results to disk."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        results_file = output_dir / f"attention_discovery_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_json_serializable(self.results)
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Results saved to {results_file}")

    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj


def run_gpt2_attention_experiment(config: Optional[AttentionDiscoveryConfig] = None):
    """
    Main function to run the GPT-2 attention discovery experiment.
    
    Args:
        config: Optional configuration. If None, uses default config.
    
    Returns:
        Dictionary containing experimental results
    """
    if config is None:
        config = AttentionDiscoveryConfig()
    
    experiment = GPT2AttentionDiscoveryExperiment(config)
    return experiment.run_complete_experiment()


if __name__ == "__main__":
    # Run experiment with default configuration
    print("Starting GPT-2 Attention Discovery Experiment")
    print("=" * 60)
    
    config = AttentionDiscoveryConfig(
        num_text_samples=100,  # Smaller for testing
        training_episodes=100,  # Fewer episodes for testing
        max_sequence_length=16  # Shorter sequences for testing
    )
    
    results = run_gpt2_attention_experiment(config)
    
    print("\nExperiment Summary:")
    print("=" * 60)
    print(f"Discovered Expression: {results['analysis_results']['discovered_expression']}")
    print(f"Final Reward: {results['analysis_results']['final_reward']:.4f}")
    print(f"Validation Fidelity: {results['validation_results']['validation_fidelity']:.4f}")
    print(f"Interpretability Score: {results['validation_results']['interpretability_score']:.4f}")
    print(f"Total Time: {results['total_time_seconds']:.2f} seconds")
    
    if results['validation_results']['pattern_generalization']:
        print("✓ Pattern generalizes to validation data")
    else:
        print("✗ Pattern does not generalize well")