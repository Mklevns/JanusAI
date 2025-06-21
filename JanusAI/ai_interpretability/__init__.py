# Init file for JanusAI.ai_interpretability module

"""
Note: This package may depend on root-level modules not yet fully integrated
into the 'JanusAI' package structure.
Ensure the project root directory is in PYTHONPATH.
"""

VERSION = "0.1.0" # Example version

# Import key components from submodules to make them available at this level
# For symbols within the ai_interpretability package, relative imports are fine.
# For symbols outside, absolute imports are used.

from JanusAI.ai_interpretability.grammars import NeuralGrammar
from JanusAI.environments.ai_interpretability.neural_net_env import (
    AIBehaviorData,
    AIInterpretabilityEnv,
    LocalInterpretabilityEnv,
    SymbolicDiscoveryEnv,
    AIDiscoveryEnv
)
from JanusAI.environments.ai_interpretability.transformer_env import TransformerInterpretabilityEnv # Assuming it's here
from JanusAI.ai_interpretability.rewards import InterpretabilityReward, FidelityRewardCalculator
from JanusAI.ai_interpretability.interpreters import AILawDiscovery
from JanusAI.ai_interpretability.symbolic.expression_parser import ExpressionParser
from JanusAI.environments.ai_interpretability.model_hooks import ModelHookManager, register_hooks_for_layers
from JanusAI.utils.visualization.plotting import ExperimentVisualizer
# 'math_utils' was too vague and its previous import '.utils' was incorrect.
# Users should import specific math utilities directly from JanusAI.utils.math

__all__ = [
    "NeuralGrammar",
    "AIBehaviorData",
    "AIInterpretabilityEnv",
    "LocalInterpretabilityEnv",
    "TransformerInterpretabilityEnv",
    "SymbolicDiscoveryEnv",
    "AIDiscoveryEnv",
    "InterpretabilityReward",
    "FidelityRewardCalculator",
    "AILawDiscovery",
    "ExpressionParser",
    "ModelHookManager",
    "register_hooks_for_layers",
    "ExperimentVisualizer",
    "VERSION",
]
