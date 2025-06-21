# JanusAI/agents/multi_agent_system.py
"""
Core Multi-Agent System for Janus scientific discovery.
Implements dynamic agent creation, role specialization, and lifecycle management.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Type, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime
import logging
from enum import Enum
import json

from janus_ai.memory.dual_memory_system import (

    DualMemorySystem, Discovery, IntermediateResult
)
from janus_ai.grammar.ai_grammar import AIGrammar

from janus_ai.environments.base.symbolic_env import TreeState

from janus_ai.rewards.interpretability_reward import InterpretabilityReward

from janus_ai.ml.training.advanced_ppo_trainer import AdvancedPPOTrainer

from janus_ai.ml.networks.policy_networks import TransformerPolicy



class AgentRole(Enum):
    """Enumeration of specialized agent roles"""
    HYPOTHESIS_GENERATOR = "hypothesis_generator"
    EXPERIMENTER = "experimenter"
    THEORIST = "theorist"
    VALIDATOR = "validator"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"
    EXPLORER = "explorer"
    REFINER = "refiner"


@dataclass
class AgentConfig:
    """Configuration for an agent"""
    role: AgentRole
    policy_config: Dict[str, Any] = field(default_factory=dict)
    specialization: Dict[str, Any] = field(default_factory=dict)
    priority: float = 1.0
    max_iterations: int = 10
    temperature: float = 1.0  # For exploration vs exploitation


class BaseScientificAgent(ABC):
    """Base class for all scientific discovery agents"""
    
    def __init__(self,
                 agent_id: str,
                 role: AgentRole,
                 policy: nn.Module,
                 grammar: AIGrammar,
                 memory_system: DualMemorySystem,
                 config: AgentConfig):
        
        self.agent_id = agent_id
        self.role = role
        self.policy = policy
        self.grammar = grammar
        self.memory_system = memory_system
        self.config = config
        
        # Agent state
        self.iteration_count = 0
        self.best_score = 0.0
        self.discovery_count = 0
        
        # Logging
        self.logger = logging.getLogger(f"{__name__}.{agent_id}")
        
    @abstractmethod
    def generate(self,
                problem_context: Dict[str, Any],
                shared_memory: List[IntermediateResult],
                episodic_memories: List[Discovery]) -> Tuple[str, str, str]:
        """
        Generate thought, response, and expression based on context.
        
        Returns:
            thought: Agent's reasoning process
            response: Natural language response
            expression: Symbolic expression
        """
        pass
    
    def act(self,
            problem_id: str,
            problem_context: Dict[str, Any],
            shared_memory: List[IntermediateResult],
            episodic_memories: List[Discovery]) -> IntermediateResult:
        """Execute agent action and return result"""
        
        self.iteration_count += 1
        
        # Generate output
        thought, response, expression = self.generate(
            problem_context, shared_memory, episodic_memories
        )
        
        # Create result
        result = IntermediateResult(
            id=f"{problem_id}_{self.agent_id}_{self.iteration_count}",
            timestamp=datetime.now(),
            domain=problem_context.get('domain', 'unknown'),
            agent_role=self.role.value,
            expression=expression,
            thought=thought,
            response=response,
            score=0.0,  # Will be set by evaluator
            iteration=self.iteration_count
        )
        
        return result
    
    def update_from_feedback(self, feedback: str, score: float):
        """Update agent based on feedback"""
        if score > self.best_score:
            self.best_score = score
            self.discovery_count += 1
            self.logger.info(f"New best score: {score:.3f}")
    
    def reset(self):
        """Reset agent state for new problem"""
        self.iteration_count = 0
        self.best_score = 0.0


class HypothesisGeneratorAgent(BaseScientificAgent):
    """Agent specialized in generating novel hypotheses"""
    
    def generate(self,
                problem_context: Dict[str, Any],
                shared_memory: List[IntermediateResult],
                episodic_memories: List[Discovery]) -> Tuple[str, str, str]:
        
        thought = "Analyzing problem space for novel hypotheses. "
        
        # Analyze episodic memories for patterns
        if episodic_memories:
            successful_patterns = [m.expression for m in episodic_memories if m.validation_score > 0.8]
            thought += f"Found {len(successful_patterns)} successful patterns from past. "
            
            # Extract common elements
            common_ops = self._extract_common_operations(successful_patterns)
            thought += f"Common operations: {', '.join(common_ops[:3])}. "
        
        # Check shared memory for current attempts
        if shared_memory:
            tried_expressions = [r.expression for r in shared_memory]
            thought += f"Already tried {len(tried_expressions)} expressions. "
            
            # Identify gaps
            untried_primitives = set(self.grammar.primitives) - set(
                op for expr in tried_expressions for op in expr.split()
            )
            thought += f"Untried primitives: {', '.join(list(untried_primitives)[:3])}. "
        
        # Generate novel hypothesis
        thought += "Generating novel combination..."
        
        # Use policy to generate expression
        with torch.no_grad():
            context_encoding = self._encode_context(
                problem_context, shared_memory, episodic_memories
            )
            
            # Higher temperature for exploration
            expression = self._sample_expression(
                context_encoding, 
                temperature=self.config.temperature * 1.2
            )
        
        response = f"Hypothesis: {expression}\n"
        response += "This expression combines "
        
        # Explain the hypothesis
        if "attention" in expression:
            response += "attention mechanisms "
        if "linear" in expression:
            response += "linear transformations "
        if "softmax" in expression:
            response += "probability normalization "
        
        response += "in a novel configuration."
        
        return thought, response, expression
    
    def _extract_common_operations(self, expressions: List[str]) -> List[str]:
        """Extract commonly used operations from successful expressions"""
        op_counts = {}
        for expr in expressions:
            for primitive in self.grammar.primitives:
                if primitive in expr:
                    op_counts[primitive] = op_counts.get(primitive, 0) + 1
        
        # Sort by frequency
        return sorted(op_counts.keys(), key=lambda x: op_counts[x], reverse=True)
    
    def _encode_context(self, problem_context, shared_memory, episodic_memories):
        """Encode all context into tensor"""
        # Simplified encoding - in practice would be more sophisticated
        features = []
        
        # Problem features
        features.append(len(shared_memory))
        features.append(len(episodic_memories))
        features.append(self.iteration_count / 10)
        
        # Performance features
        if shared_memory:
            features.append(max(r.score for r in shared_memory))
        else:
            features.append(0.0)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _sample_expression(self, context: torch.Tensor, temperature: float) -> str:
        """Sample expression from policy"""
        # Simplified - in practice would use actual policy sampling
        if np.random.random() < 0.7:  # 70% chance of building on existing
            base_ops = ["softmax", "attention", "linear", "scale"]
            selected_ops = np.random.choice(base_ops, size=2, replace=False)
            return f"{selected_ops[0]}({selected_ops[1]}(Q, K) / sqrt(d))"
        else:  # 30% chance of trying something new
            new_ops = ["log_softmax", "layer_norm", "position_encoding", "mask"]
            selected_op = np.random.choice(new_ops)
            return f"{selected_op}(attention(Q, K))"


class ExperimenterAgent(BaseScientificAgent):
    """Agent specialized in designing and running experiments"""
    
    def generate(self,
                problem_context: Dict[str, Any],
                shared_memory: List[IntermediateResult],
                episodic_memories: List[Discovery]) -> Tuple[str, str, str]:
        
        thought = "Designing experiment to test current hypotheses. "
        
        # Get best hypothesis from shared memory
        if shared_memory:
            best_hypothesis = max(shared_memory, key=lambda r: r.score)
            thought += f"Testing: {best_hypothesis.expression}. "
            
            # Design experiment based on hypothesis
            if "attention" in best_hypothesis.expression:
                thought += "Creating attention pattern test cases. "
                expression = self._design_attention_experiment(best_hypothesis.expression)
            else:
                thought += "Creating general function test cases. "
                expression = self._design_general_experiment(best_hypothesis.expression)
        else:
            thought += "No hypothesis to test yet. Proposing exploratory experiment. "
            expression = "test_random_patterns()"
        
        response = f"Experiment: Test {expression} under various conditions\n"
        response += "Test conditions:\n"
        response += "- Sequence lengths: [8, 16, 32]\n"
        response += "- Noise levels: [0.0, 0.1, 0.2]\n"
        response += "- Edge cases: empty, single element, maximum size"
        
        return thought, response, expression
    
    def _design_attention_experiment(self, hypothesis: str) -> str:
        """Design experiment for attention hypothesis"""
        test_patterns = ["diagonal", "previous_token", "uniform", "random"]
        selected = np.random.choice(test_patterns, size=2)
        return f"test({hypothesis}, patterns={selected})"
    
    def _design_general_experiment(self, hypothesis: str) -> str:
        """Design general experiment"""
        return f"validate({hypothesis}, samples=100)"


class TheoristAgent(BaseScientificAgent):
    """Agent specialized in theoretical analysis and synthesis"""
    
    def generate(self,
                problem_context: Dict[str, Any],
                shared_memory: List[IntermediateResult],
                episodic_memories: List[Discovery]) -> Tuple[str, str, str]:
        
        thought = "Analyzing discoveries for theoretical insights. "
        
        # Analyze shared memory for patterns
        if shared_memory:
            expressions = [r.expression for r in shared_memory]
            scores = [r.score for r in shared_memory]
            
            thought += f"Analyzing {len(expressions)} expressions. "
            
            # Find commonalities in high-scoring expressions
            high_scoring = [expr for expr, score in zip(expressions, scores) if score > 0.7]
            if high_scoring:
                thought += f"Found {len(high_scoring)} promising approaches. "
                
                # Synthesize theory
                common_structure = self._find_common_structure(high_scoring)
                thought += f"Common structure: {common_structure}. "
                
                expression = self._synthesize_theory(common_structure, high_scoring)
            else:
                thought += "No high-scoring expressions yet. Proposing theoretical framework. "
                expression = "general_attention_framework(Q, K, V)"
        else:
            thought += "No data yet. Proposing initial theoretical framework. "
            expression = "base_theory()"
        
        response = f"Theoretical Analysis:\n"
        response += f"Proposed unified framework: {expression}\n"
        response += "This framework captures the essential structure while allowing variations."
        
        return thought, response, expression
    
    def _find_common_structure(self, expressions: List[str]) -> str:
        """Find common structural elements"""
        # Simplified - in practice would use tree analysis
        if all("softmax" in expr for expr in expressions):
            if all("attention" in expr for expr in expressions):
                return "softmax(attention(...))"
            return "softmax(...)"
        return "unknown"
    
    def _synthesize_theory(self, structure: str, examples: List[str]) -> str:
        """Synthesize theoretical framework"""
        if "softmax(attention" in structure:
            return "generalized_attention(Q, K, V, normalize=softmax)"
        return f"theoretical_framework({structure})"


class ValidatorAgent(BaseScientificAgent):
    """Agent specialized in validating discoveries"""
    
    def generate(self,
                problem_context: Dict[str, Any],
                shared_memory: List[IntermediateResult],
                episodic_memories: List[Discovery]) -> Tuple[str, str, str]:
        
        thought = "Validating current best hypothesis. "
        
        if shared_memory:
            # Get top hypothesis
            top_result = max(shared_memory, key=lambda r: r.score)
            thought += f"Validating: {top_result.expression}. "
            
            # Perform validation checks
            checks = self._perform_validation_checks(top_result.expression, problem_context)
            
            thought += f"Performed {len(checks)} validation checks. "
            passed = sum(1 for check in checks.values() if check)
            thought += f"{passed}/{len(checks)} checks passed. "
            
            # Create validation expression
            if passed == len(checks):
                expression = f"validated({top_result.expression})"
            else:
                expression = f"needs_refinement({top_result.expression})"
        else:
            thought += "No hypothesis to validate yet. "
            expression = "awaiting_hypothesis()"
        
        response = "Validation Report:\n"
        if shared_memory:
            for check_name, passed in checks.items():
                status = "✓" if passed else "✗"
                response += f"{status} {check_name}\n"
        else:
            response += "Waiting for hypotheses to validate."
        
        return thought, response, expression
    
    def _perform_validation_checks(self, expression: str, context: Dict[str, Any]) -> Dict[str, bool]:
        """Perform various validation checks"""
        checks = {}
        
        # Syntax check
        checks["valid_syntax"] = self._check_syntax(expression)
        
        # Complexity check
        checks["reasonable_complexity"] = len(expression) < 100
        
        # Domain appropriateness
        if context.get('domain') == 'attention_mechanisms':
            checks["contains_attention_ops"] = any(
                op in expression for op in ["attention", "softmax", "Q", "K"]
            )
        
        # Mathematical properties
        checks["mathematically_sound"] = "divide_by_zero" not in expression
        
        return checks
    
    def _check_syntax(self, expression: str) -> bool:
        """Check if expression has valid syntax"""
        # Simple parenthesis matching
        count = 0
        for char in expression:
            if char == '(':
                count += 1
            elif char == ')':
                count -= 1
            if count < 0:
                return False
        return count == 0


class CriticAgent(BaseScientificAgent):
    """Agent specialized in constructive criticism and improvement suggestions"""
    
    def generate(self,
                problem_context: Dict[str, Any],
                shared_memory: List[IntermediateResult],
                episodic_memories: List[Discovery]) -> Tuple[str, str, str]:
        
        thought = "Analyzing current approaches for improvements. "
        
        critiques = []
        
        if shared_memory:
            # Analyze each result
            for result in shared_memory[-3:]:  # Last 3 attempts
                critique = self._critique_expression(result.expression, result.score)
                critiques.append(critique)
                thought += f"Critiqued {result.agent_role}: {critique['issue']}. "
            
            # Suggest improvement
            improvement = self._suggest_improvement(shared_memory, critiques)
            expression = improvement
        else:
            thought += "No attempts to critique yet. "
            expression = "needs_initial_hypothesis()"
        
        response = "Critical Analysis:\n"
        for i, critique in enumerate(critiques):
            response += f"\n{i+1}. {critique['expression']}:\n"
            response += f"   Issue: {critique['issue']}\n"
            response += f"   Suggestion: {critique['suggestion']}\n"
        
        if expression != "needs_initial_hypothesis()":
            response += f"\nRecommended improvement: {expression}"
        
        return thought, response, expression
    
    def _critique_expression(self, expression: str, score: float) -> Dict[str, str]:
        """Provide constructive criticism of expression"""
        critique = {
            'expression': expression,
            'issue': '',
            'suggestion': ''
        }
        
        if score < 0.5:
            critique['issue'] = "Low performance score"
            critique['suggestion'] = "Consider fundamental restructuring"
        elif len(expression) > 80:
            critique['issue'] = "Overly complex"
            critique['suggestion'] = "Simplify by removing redundant operations"
        elif "scale" not in expression and "sqrt" not in expression:
            critique['issue'] = "Missing scaling factor"
            critique['suggestion'] = "Add scaling by sqrt(d) for stability"
        else:
            critique['issue'] = "Minor optimization possible"
            critique['suggestion'] = "Consider parameter tuning"
        
        return critique
    
    def _suggest_improvement(self, shared_memory: List[IntermediateResult], 
                           critiques: List[Dict]) -> str:
        """Suggest concrete improvement based on critiques"""
        # Get best expression
        best = max(shared_memory, key=lambda r: r.score)
        
        # Apply most common suggestion
        if any("scaling" in c['suggestion'] for c in critiques):
            return best.expression.replace(")", " / sqrt(d))")
        elif any("Simplify" in c['suggestion'] for c in critiques):
            # Remove nested operations
            return best.expression.replace("softmax(softmax(", "softmax(")
        else:
            return f"refined({best.expression})"


class DynamicAgentPool:
    """Manages a pool of agents dynamically"""
    
    def __init__(self,
                 memory_system: DualMemorySystem,
                 grammar: AIGrammar,
                 base_policy: nn.Module,
                 max_agents: int = 10):
        
        self.memory_system = memory_system
        self.grammar = grammar
        self.base_policy = base_policy
        self.max_agents = max_agents
        
        # Agent registry
        self.agent_classes = {
            AgentRole.HYPOTHESIS_GENERATOR: HypothesisGeneratorAgent,
            AgentRole.EXPERIMENTER: ExperimenterAgent,
            AgentRole.THEORIST: TheoristAgent,
            AgentRole.VALIDATOR: ValidatorAgent,
            AgentRole.CRITIC: CriticAge
