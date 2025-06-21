# # janus/experiments/configs/run_gpt_2_attention_experiment.py

# """
# GPT-2 Attention Discovery Experiment
# This module implements a complete experiment to discover symbolic patterns
# in the attention heads of a GPT-2 model using PPO or symbolic/genetic methods.
# It includes setup, training, and evaluation phases.
# """

import json
import time
import logging

import numpy as np
import torch
from transformers import GPT2Model, GPT2Tokenizer
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from janus_ai.experiments.registry import register_experiment
from janus_ai.experiments.base import BaseExperiment
from janus_ai.experiments.config import GPT2AttentionConfig as RegistryConfig
from janus_ai.ai_interpretability.evaluation.fidelity import FidelityCalculator
from janus_ai.core.grammar import EnhancedAIGrammar
from janus_ai.ai_interpretability.rewards.interpretability_reward import InterpretabilityReward
from janus_ai.environments.base.symbolic_env import SymbolicDiscoveryEnv
from janus_ai.ml.training.ppo_trainer import PPOTrainer

# Device for computation
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class GPT2AttentionConfig:
    model_name: str = 'gpt2'
    layer_index: int = 0
    head_index: Optional[int] = 1  # previous-token head
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

@register_experiment(
    name="gpt2_attention_discovery",
    category="ai",
    aliases=["transformer_attention", "attention_patterns"],
    description="Discover symbolic patterns in GPT-2 attention heads",
    tags=["nlp", "transformer", "attention", "interpretability"],
    supported_algorithms=["ppo", "symbolic_regression", "genetic"],
    config_schema={
        'model_name': {'type': str, 'default': 'gpt2'},
        'layer_index': {'type': int, 'default': 0, 'choices': list(range(12))},
        'head_index': {'type': int, 'default': 1, 'choices': [None] + list(range(12))},
        'max_complexity': {'type': int, 'default': 20},
        'num_text_samples': {'type': int, 'default': 100},
        'max_sequence_length': {'type': int, 'default': 64},
        'training_episodes': {'type': int, 'default': 200},
        'learning_rate': {'type': float, 'default': 3e-4},
        'gamma': {'type': float, 'default': 0.99}
    }
)
class GPT2AttentionExperiment(BaseExperiment):
    """
    Complete GPT-2 attention head discovery experiment using PPO or symbolic/genetic.
    """
    def __init__(self, config: GPT2AttentionConfig):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.model: GPT2Model = None
        self.tokenizer: GPT2Tokenizer = None
        self.attention_data: Dict[str, List[Any]] = {}
        self.variables: List[Any] = []
        self.grammar: EnhancedAIGrammar = None
        self.reward_calculator: InterpretabilityReward = None
        self.fidelity_calculator: FidelityCalculator = None
        self.environment: SymbolicDiscoveryEnv = None
        self.trainer: PPOTrainer = None
        self.training_history: List[Dict[str, Any]] = []

    def setup(self):
        self.logger.info("Setting up GPT-2 attention discovery...")
        self._load_model()
        self._extract_attention_data()
        self._create_variables()
        self._setup_grammar()
        self._setup_reward_calculator()
        self._create_environment()
        self._setup_trainer()
        self.logger.info("Setup complete.")

    def _load_model(self):
        self.logger.info(f"Loading GPT-2 model '{self.config.model_name}'...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2Model.from_pretrained(
            self.config.model_name, output_attentions=True
        ).to(DEVICE).eval()
        self.logger.info(f"Model loaded to {DEVICE}")

    def _extract_attention_data(self):
        self.logger.info("Extracting attention patterns...")
        data = {'attention_weights': [], 'sequence_lengths': [], 'token_types': []}
        for text in self._generate_text_samples():
            enc = self.tokenizer(
                text,
                return_tensors='pt', padding=True,
                truncation=True, max_length=self.config.max_sequence_length
            )
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model(**enc)
            attn = out.attentions[self.config.layer_index]
            head = attn[0, self.config.head_index] if self.config.head_index is not None else attn.mean(dim=1)[0]
            arr = head.cpu().numpy()
            seq = enc['input_ids'].shape[1]
            tokens = self.tokenizer.convert_ids_to_tokens(enc['input_ids'][0])
            types = [1 if t.startswith('Ġ') else 0 for t in tokens]
            data['attention_weights'].append(arr)
            data['sequence_lengths'].append(seq)
            data['token_types'].append(np.array(types))
        self.attention_data = data
        self.logger.info(f"Collected {len(data['attention_weights'])} samples.")

    def _generate_text_samples(self) -> List[str]:
        base = [
            "The cat sat on the mat.", "A B A B A B A B", "hello world hello world",
            "The quick brown fox jumps over the lazy dog.",
            "abc abc abc abc", "red blue red blue red blue"
        ]
        return [
            base[i % len(base)].upper() if i % 2 else base[i % len(base)].lower()
            for i in range(self.config.num_text_samples)
        ]

    def _create_variables(self):
        from dataclasses import dataclass
        @dataclass
        class Variable:
            name: str; index: int; properties: Dict[str, Any]
        self.variables = [
            Variable('pos_diff',0,{}), Variable('pos_ratio',1,{}),
            Variable('token_type_i',2,{}), Variable('token_type_j',3,{}),
            Variable('relative_pos',4,{}), Variable('is_previous',5,{})
        ]
        self.logger.info(f"Created {len(self.variables)} variables.")

    def _setup_grammar(self):
        self.grammar = EnhancedAIGrammar()
        self.grammar.add_custom_primitive(
            'is_previous_token', lambda i, j: (i-j)==1, 'pattern'
        )
        self.logger.info("Grammar ready.")

    def _setup_reward_calculator(self):
        self.reward_calculator = InterpretabilityReward(
            reward_weights=self.config.reward_weights,
            complexity_penalty_factor=0.01
        )
        self.reward_calculator.variables = self.variables
        self.fidelity_calculator = FidelityCalculator()
        self.logger.info("Reward calculator ready.")

    def _create_environment(self):
        ins, outs = [], []
        for mat, seq, types in zip(
            self.attention_data['attention_weights'],
            self.attention_data['sequence_lengths'],
            self.attention_data['token_types']):
            for i in range(seq):
                for j in range(seq):
                    ins.append([i-j, i/max(1,j), types[i], types[j], abs(i-j), float(i-j==1)])
                    outs.append(mat[i,j])
        data_dict = {'inputs':np.array(ins),'outputs':np.array(outs)}
        self.environment = SymbolicDiscoveryEnv(
            grammar=self.grammar,
            target_data=data_dict,
            variables=self.variables,
            reward_calculator=self.reward_calculator,
            device=DEVICE
        )
        self.logger.info("Environment created.")

    def _setup_trainer(self):
        self.trainer = PPOTrainer(
            env=self.environment,
            learning_rate=self.config.learning_rate,
            gamma=self.config.gamma,
            device=DEVICE
        )
        self.logger.info("PPO Trainer initialized.")

    def run(self) -> Any:
        self.logger.info("Starting discovery...")
        history = self.trainer.train(num_episodes=self.config.training_episodes)
        best_expr, best_reward = self.trainer.get_best_solution()
        self.training_history = history
        return best_expr

    def teardown(self):
        self.logger.info("Cleaning up...")
        del self.model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def postprocess(self, result) -> Dict[str, Any]:
        return {
            'discovered_expression': str(result),
            'config': self.config.__dict__,
            'training_history': self.training_history
        }

# If run directly, fallback
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    cfg = GPT2AttentionConfig()
    exp = GPT2AttentionExperiment(cfg)
    exp.setup()
    res = exp.run()
    out = exp.postprocess(res)
    print(json.dumps(out, indent=2))
