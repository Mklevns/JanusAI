import json
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import numpy as np
import torch
from transformers import GPT2Model, GPT2Tokenizer

# Janus imports
from janus_ai.ai_interpretability.evaluation.fidelity import FidelityCalculator
from janus_ai.core.grammar import EnhancedAIGrammar
from janus_ai.ai_interpretability.rewards.interpretability_reward import InterpretabilityReward
from janus_ai.environments.base.symbolic_env import SymbolicDiscoveryEnv
from janus_ai.ml.training.ppo_trainer import PPOTrainer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
\@dataclass
class GPT2AttentionConfig:
    model_name: str = 'gpt2'
    layer_index: int = 0
    head_index: int = 1         # previous-token head
    max_complexity: int = 20
    num_text_samples: int = 100
    max_sequence_length: int = 64
    training_episodes: int = 200
    learning_rate: float = 3e-4
    gamma: float = 0.99
    reward_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.reward_weights is None:
            self.reward_weights = {
                'fidelity': 0.6,
                'simplicity': 0.2,
                'consistency': 0.1,
                'insight': 0.1
            }


class GPT2AttentionExperiment:
    def __init__(self, config: GPT2AttentionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # GPT-2 components
        self.model: GPT2Model = None
        self.tokenizer: GPT2Tokenizer = None

        # Data
        self.attention_data: Dict[str, List[Any]] = {}
        self.variables: List[Any] = []

        # Core modules
        self.grammar: EnhancedAIGrammar = None
        self.reward_calculator: InterpretabilityReward = None
        self.fidelity_calculator: FidelityCalculator = None
        self.environment: SymbolicDiscoveryEnv = None
        self.trainer: PPOTrainer = None

        self.training_history: List[Dict[str, Any]] = []

    def setup(self):
        self.logger.info("Setting up experiment...")
        self._load_model()
        self._extract_attention_data()
        self._create_variables()
        self._setup_grammar()
        self._setup_reward_calculator()
        self._create_environment()
        self._setup_trainer()
        self.logger.info("Setup complete.")

    def _load_model(self):
        self.logger.info(f"Loading GPT2 model '{self.config.model_name}'...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2Model.from_pretrained(
            self.config.model_name,
            output_attentions=True
        ).to(device).eval()
        self.logger.info(f"Model loaded to {device}")

    def _generate_text_samples(self) -> List[str]:
        base = [
            "The cat sat on the mat.", "A B A B A B A B", "hello world hello world",
            "The quick brown fox jumps over the lazy dog.",
            "abc abc abc abc", "red blue red blue red blue"
        ]
        samples = []
        for i in range(self.config.num_text_samples):
            s = base[i % len(base)]
            samples.append(s.upper() if i % 2 else s.lower())
        return samples

    def _extract_attention_data(self):
        self.logger.info("Extracting attention patterns...")
        data = {'input_ids': [], 'attention_weights': [], 'sequence_lengths': [], 'token_types': []}
        for text in self._generate_text_samples():
            enc = self.tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.config.max_sequence_length
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model(**enc)
            attn = out.attentions[self.config.layer_index][0, self.config.head_index]
            arr = attn.cpu().numpy()
            seq = enc['input_ids'].shape[1]
            data['input_ids'].append(enc['input_ids'][0].cpu().numpy())
            data['attention_weights'].append(arr)
            data['sequence_lengths'].append(seq)
            tokens = self.tokenizer.convert_ids_to_tokens(enc['input_ids'][0])
            types = [1 if t.startswith('Ġ') else 0 for t in tokens]
            data['token_types'].append(np.array(types))
        self.attention_data = data
        self.logger.info(f"Collected {len(data['attention_weights'])} samples.")

    def _create_variables(self):
        from dataclasses import dataclass
        @dataclass
        class Variable:
            name: str; index: int; properties: Dict[str, Any]
        self.variables = [
            Variable('pos_diff', 0, {}),
            Variable('pos_ratio',1, {}),
            Variable('token_type_i',2,{}),
            Variable('token_type_j',3,{}),
            Variable('relative_pos',4,{}),
            Variable('is_previous',5,{})
        ]
        self.logger.info(f"Created {len(self.variables)} variables.")

    def _setup_grammar(self):
        self.grammar = EnhancedAIGrammar()
        self.grammar.add_custom_primitive(
            'is_previous_token', lambda i, j: (i - j) == 1, 'pattern'
        )
        self.logger.info("Grammar initialized.")

    def _setup_reward_calculator(self):
        self.reward_calculator = InterpretabilityReward(
            reward_weights=self.config.reward_weights,
            complexity_penalty_factor=0.01
        )
        self.reward_calculator.variables = self.variables
        self.fidelity_calculator = FidelityCalculator()
        self.logger.info("Reward calculator ready.")

    def _create_environment(self):
        # Prepare data arrays
        ins, outs = [], []
        for mat, seq, types in zip(
            self.attention_data['attention_weights'],
            self.attention_data['sequence_lengths'],
            self.attention_data['token_types']):
            for i in range(seq):
                for j in range(seq):
                    ins.append([i - j,
                                i / max(1, j),
                                types[i], types[j],
                                abs(i - j), float(i - j == 1)])
                    outs.append(mat[i, j])
        data_dict = {
            'inputs': np.array(ins),
            'outputs': np.array(outs),
            'sequence_length': self.config.max_sequence_length,
            'token_types': None
        }
        self.environment = SymbolicDiscoveryEnv(
            grammar=self.grammar,
            target_data=data_dict,
            variables=self.variables,
            reward_calculator=self.reward_calculator,
            device=device
        )
        self.logger.info("Environment created with GPU support." )

    def _setup_trainer(self):
        # Use PPO for policy optimization
        self.trainer = PPOTrainer(
            env=self.environment,
            learning_rate=self.config.learning_rate,
            gamma=self.config.gamma,
            device=device
        )
        self.logger.info("PPO Trainer initialized.")

    def run_discovery(self) -> Dict[str, Any]:
        self.logger.info("Starting discovery with PPO...")
        # Train policy
        history = self.trainer.train(
            num_episodes=self.config.training_episodes
        )
        # Retrieve best
        best_expr, best_reward = self.trainer.get_best_solution()
        self.training_history = history
        return {
            'expression': best_expr,
            'reward': best_reward,
            'history': history
        }

    def save_results(self, results: Dict[str, Any], path: str = "results.json"):
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Saved results to {path}")


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    cfg = GPT2AttentionConfig()
    exp = GPT2AttentionExperiment(cfg)
    exp.setup()
    results = exp.run_discovery()
    exp.save_results(results)
    print("Discovery complete:\n", results)

if __name__ == '__main__':
    main()
