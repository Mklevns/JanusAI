"""
Progressive Grammar System for Autonomous Physics Discovery
==========================================================

A hierarchical grammar that discovers variables from observations and
progressively builds mathematical abstractions using information-theoretic principles.

This module now serves as a facade, re-exporting classes from their new locations
to maintain backward compatibility.
"""

import warnings

# Re-export classes from their new locations
from janus_ai.core.grammar.cfg import CFGRule, ContextFreeGrammar
from janus_ai.core.grammar.denoiser import NoisyObservationProcessor
from janus_ai.core.grammar.progressive_grammar import ProgressiveGrammar
from janus_ai.core.grammar.ai_grammar import AIGrammar

# It's also common to re-export key components from sibling modules if they were
# previously accessible via this base module, e.g., Expression and Variable.
# Assuming Expression and Variable were commonly imported alongside grammar classes:
from janus_ai.core.expressions.expression import Expression, Variable # Adjusted path

# And the TargetType alias if it was considered part of this module's public API
from janus_ai.core.grammar.cfg import TargetType # TargetType is defined and used in cfg.py

warnings.filterwarnings('ignore')

__all__ = [
    "CFGRule",
    "ContextFreeGrammar",
    "NoisyObservationProcessor",
    "ProgressiveGrammar",
    "AIGrammar",
    "Expression",
    "Variable",
    "TargetType",
]
