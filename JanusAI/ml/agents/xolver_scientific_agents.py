# JanusAI/ml/agents/xolver_scientific_agents.py
"""
Xolver-inspired multi-agent system for Janus scientific discovery.
Implements specialized agents with dual memory and judge-mediated refinement.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
from abc import ABC, abstractmethod
import json

from janus_ai.grammar.ai_grammar import AIGrammar


@dataclass
class ScientificDiscovery:
    """Represents a scientific discovery with full context"""
    expression: str
    domain: str
    hypothesis: str
    evidence: List[Dict[str, Any]]
    confidence: float
    reasoning_trace: List[str]
    agent_roles: List[str]
    iteration_count: int
    validation_score: float = 0.0
    
    def to_episodic_entry(self) -> Dict[str, Any]:
        """Convert to format for episodic memory storage"""
        return {
            'expression': self.expression,
            'domain': self.domain,
            'hypothesis': self.hypothesis,
            'evidence': self.evidence,
            'confidence': self.confidence,
            'reasoning_trace': self.reasoning_trace,
            'validation_score': self.validation_score,
            'metadata': {
                'agent_roles': self.agent_roles,
                'iteration_count': self.iteration_count
            }
        }


class ScientificAgent(ABC):
    """Base class for specialized scientific discovery agents"""
    
    def __init__(self, role: str, model: torch.nn.Module):
        self.role = role
        self.model = model
        self.experience_buffer = deque(maxlen=100)
        
    @abstractmethod
    def generate(self, 
                context: Dict[str, Any],
                shared_memory: List[Dict[str, Any]],
                episodic_memory: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, str]:
        """Generate thought and response given context and memories"""
        pass
    
    def update_experience(self, experience: Dict[str, Any]):
        """Update agent's individual experience buffer"""
        self.experience_buffer.append(experience)


class HypothesisGeneratorAgent(ScientificAgent):
    """Agent specialized in generating novel hypotheses"""
    
    def __init__(self, model: torch.nn.Module, grammar: AIGrammar):
        super().__init__("Hypothesis Generator", model)
        self.grammar = grammar
        
    def generate(self, 
                context: Dict[str, Any],
                shared_memory: List[Dict[str, Any]],
                episodic_memory: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, str]:
        
        # Extract problem context
        target_data = context.get('target_data')
        domain = context.get('domain', 'attention_mechanisms')
        
        # Build prompt incorporating memories
        thought = f"Analyzing {domain} patterns. "
        
        # Learn from episodic memory
        if episodic_memory:
            similar_discoveries = self._find_similar_discoveries(target_data, episodic_memory)
            if similar_discoveries:
                thought += f"Found {len(similar_discoveries)} related discoveries. "
                thought += "Building on previous insights: "
                for disc in similar_discoveries[:3]:
                    thought += f"{disc['expression']} (confidence: {disc['confidence']:.2f}), "
        
        # Learn from shared memory (current problem)
        if shared_memory:
            best_current = max(shared_memory, key=lambda x: x.get('score', 0))
            thought += f"\nCurrent best approach: {best_current.get('expression', 'none')} "
            thought += f"with score {best_current.get('score', 0):.2f}. "
        
        # Generate novel hypothesis
        thought += "\nProposing novel symbolic expression..."
        
        # Use model to generate expression
        with torch.no_grad():
            # Encode context
            context_encoding = self._encode_context(context, shared_memory, episodic_memory)
            
            # Generate expression tree
            expression_actions = self.model.generate_expression(
                context_encoding,
                self.grammar,
                temperature=1.2  # Higher for exploration
            )
            
        expression = self.grammar.actions_to_expression(expression_actions)
        
        response = f"Hypothesis: {expression}\n"
        response += "This expression captures the pattern by combining "
        response += f"{self._describe_expression_components(expression)}"
        
        return thought, response
    
    def _find_similar_discoveries(self, target_data: Any, 
                                 episodic_memory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find discoveries relevant to current problem"""
        # Simplified similarity - in practice would use embeddings
        return [d for d in episodic_memory if d.get('validation_score', 0) > 0.8][:5]
    
    def _encode_context(self, context: Dict[str, Any], 
                       shared_memory: List[Dict[str, Any]],
                       episodic_memory: Optional[List[Dict[str, Any]]]) -> torch.Tensor:
        """Encode all context into tensor for model"""
        features = []
        
        # Problem features
        features.extend([
            len(shared_memory),
            max([m.get('score', 0) for m in shared_memory], default=0),
            len(episodic_memory) if episodic_memory else 0
        ])
        
        # Add more domain-specific features
        return torch.tensor(features, dtype=torch.float32)
    
    def _describe_expression_components(self, expression: str) -> str:
        """Generate natural language description of expression components"""
        components = []
        if 'attention' in expression:
            components.append("attention mechanisms")
        if 'softmax' in expression:
            components.append("probability normalization")
        if 'linear' in expression:
            components.append("linear transformations")
        return ", ".join(components) if components else "basic operations"


class ExperimentDesignerAgent(ScientificAgent):
    """Agent specialized in designing experiments to test hypotheses"""
    
    def generate(self, 
                context: Dict[str, Any],
                shared_memory: List[Dict[str, Any]],
                episodic_memory: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, str]:
        
        hypothesis = context.get('hypothesis', '')
        target_pattern = context.get('target_pattern')
        
        thought = f"Designing experiment to test: {hypothesis}\n"
        thought += "Need to create test cases that isolate key behaviors..."
        
        # Design experiment based on hypothesis structure
        if 'attention' in hypothesis:
            test_design = {
                'type': 'attention_pattern_matching',
                'sequence_lengths': [8, 16, 32],
                'pattern_types': ['diagonal', 'previous_token', 'local_window'],
                'noise_levels': [0.0, 0.1, 0.2]
            }
        else:
            test_design = {
                'type': 'general_function_fitting',
                'input_ranges': [(-2, 2), (-5, 5)],
                'sample_sizes': [50, 100, 200]
            }
        
        response = f"Experiment Design:\n{json.dumps(test_design, indent=2)}\n"
        response += "This will test the hypothesis under various conditions "
        response += "to verify its generalization and robustness."
        
        return thought, response


class SymbolicReasonerAgent(ScientificAgent):
    """Agent specialized in symbolic manipulation and simplification"""
    
    def __init__(self, model: torch.nn.Module, grammar: AIGrammar):
        super().__init__("Symbolic Reasoner", model)
        self.grammar = grammar
        
    def generate(self, 
                context: Dict[str, Any],
                shared_memory: List[Dict[str, Any]],
                episodic_memory: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, str]:
        
        current_expression = context.get('expression', '')
        
        thought = f"Analyzing expression: {current_expression}\n"
        thought += "Looking for simplifications and mathematical properties..."
        
        # Analyze expression structure
        simplified = self._attempt_simplification(current_expression)
        properties = self._analyze_properties(current_expression)
        
        response = f"Simplified form: {simplified}\n"
        response += f"Properties: {', '.join(properties)}\n"
        
        # Suggest improvements based on shared memory
        if shared_memory:
            successful_patterns = [m['expression'] for m in shared_memory if m.get('score', 0) > 0.8]
            if successful_patterns:
                response += "\nSuccessful patterns from other agents:\n"
                for pattern in successful_patterns[:3]:
                    response += f"- {pattern}\n"
        
        return thought, response
    
    def _attempt_simplification(self, expression: str) -> str:
        """Attempt to simplify the expression"""
        # Placeholder - would use symbolic math library
        if 'softmax(attention(' in expression and 'scale' not in expression:
            return expression.replace('attention(', 'attention(scale=1/sqrt(d), ')
        return expression
    
    def _analyze_properties(self, expression: str) -> List[str]:
        """Analyze mathematical properties of expression"""
        properties = []
        if 'softmax' in expression:
            properties.append("probabilistic (sums to 1)")
        if 'attention' in expression:
            properties.append("permutation equivariant")
        if 'linear' in expression:
            properties.append("linear transformation")
        return properties


class ValidationAgent(ScientificAgent):
    """Agent specialized in validating discoveries against data"""
    
    def generate(self, 
                context: Dict[str, Any],
                shared_memory: List[Dict[str, Any]],
                episodic_memory: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, str]:
        
        expression = context.get('expression', '')
        test_data = context.get('test_data')
        
        thought = f"Validating expression: {expression}\n"
        thought += "Running comprehensive tests on held-out data..."
        
        # Simulate validation (in practice would compile and execute)
        validation_results = {
            'accuracy': np.random.uniform(0.7, 0.95),
            'consistency': np.random.uniform(0.8, 1.0),
            'generalization': np.random.uniform(0.6, 0.9)
        }
        
        response = f"Validation Results:\n"
        response += f"- Accuracy: {validation_results['accuracy']:.3f}\n"
        response += f"- Consistency: {validation_results['consistency']:.3f}\n"
        response += f"- Generalization: {validation_results['generalization']:.3f}\n"
        
        overall_score = np.mean(list(validation_results.values()))
        response += f"\nOverall validation score: {overall_score:.3f}"
        
        return thought, response


class JudgeAgent:
    """Judge agent that evaluates and ranks agent outputs"""
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
        
    def evaluate(self, 
                agent_outputs: List[Tuple[str, str, str, str]],  # (agent_role, thought, response, expression)
                context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate and score agent outputs"""
        
        evaluations = []
        
        for role, thought, response, expression in agent_outputs:
            # Evaluate based on multiple criteria
            scores = self._compute_scores(expression, thought, response, context)
            
            feedback = self._generate_feedback(scores, expression)
            
            evaluations.append({
                'agent_role': role,
                'thought': thought,
                'response': response,
                'expression': expression,
                'score': scores['overall'],
                'detailed_scores': scores,
                'feedback': feedback
            })
        
        # Sort by overall score
        evaluations.sort(key=lambda x: x['score'], reverse=True)
        
        return evaluations
    
    def _compute_scores(self, expression: str, thought: str, 
                       response: str, context: Dict[str, Any]) -> Dict[str, float]:
        """Compute multi-dimensional scores"""
        
        scores = {
            'correctness': self._score_correctness(expression, context),
            'novelty': self._score_novelty(expression, context),
            'simplicity': self._score_simplicity(expression),
            'reasoning_quality': self._score_reasoning(thought, response)
        }
        
        # Weighted average
        weights = {'correctness': 0.4, 'novelty': 0.2, 'simplicity': 0.2, 'reasoning_quality': 0.2}
        scores['overall'] = sum(scores[k] * weights[k] for k in weights)
        
        return scores
    
    def _score_correctness(self, expression: str, context: Dict[str, Any]) -> float:
        """Score how well expression matches target"""
        # Placeholder - would compute actual fidelity
        return np.random.uniform(0.6, 1.0)
    
    def _score_novelty(self, expression: str, context: Dict[str, Any]) -> float:
        """Score novelty compared to previous attempts"""
        shared_memory = context.get('shared_memory', [])
        existing_expressions = [m.get('expression', '') for m in shared_memory]
        
        if expression in existing_expressions:
            return 0.1
        
        # Check structural similarity
        return np.random.uniform(0.5, 1.0)
    
    def _score_simplicity(self, expression: str) -> float:
        """Score based on expression complexity"""
        # Simple length-based metric
        length_score = 1.0 / (1.0 + len(expression) / 50)
        
        # Penalize deeply nested structures
        nesting_penalty = expression.count('(') / 10
        
        return max(0.1, length_score - nesting_penalty)
    
    def _score_reasoning(self, thought: str, response: str) -> float:
        """Score quality of reasoning"""
        # Check for key reasoning indicators
        indicators = ['because', 'therefore', 'building on', 'evidence', 'validates']
        indicator_count = sum(1 for ind in indicators if ind in thought.lower() or ind in response.lower())
        
        return min(1.0, 0.5 + indicator_count * 0.1)
    
    def _generate_feedback(self, scores: Dict[str, float], expression: str) -> str:
        """Generate constructive feedback"""
        feedback = []
        
        if scores['correctness'] < 0.7:
            feedback.append("Expression needs better alignment with target pattern")
        
        if scores['novelty'] < 0.5:
            feedback.append("Try exploring more diverse expression structures")
            
        if scores['simplicity'] < 0.5:
            feedback.append(f"Expression is complex ({len(expression)} chars), consider simplification")
            
        if scores['reasoning_quality'] < 0.7:
            feedback.append("Strengthen reasoning with more evidence and logical connections")
        
        return "; ".join(feedback) if feedback else "Good work! Consider minor refinements."


class XolverScientificDiscoverySystem:
    """
    Complete Xolver-inspired system for Janus scientific discovery.
    Implements dual memory, dynamic agents, and iterative refinement.
    """
    
    def __init__(self,
                 base_model: torch.nn.Module,
                 grammar: AIGrammar,
                 num_agents: int = 3,
                 max_iterations: int = 3):
        
        self.grammar = grammar
        self.num_agents = num_agents
        self.max_iterations = max_iterations
        
        # Initialize specialized agents
        self.agent_types = {
            'hypothesis_generator': HypothesisGeneratorAgent,
            'experiment_designer': ExperimentDesignerAgent,
            'symbolic_reasoner': SymbolicReasonerAgent,
            'validator': ValidationAgent
        }
        
        # Dual memory system
        self.episodic_memory = deque(maxlen=10000)  # Long-term discoveries
        self.shared_memory = []  # Per-problem intermediate memory
        
        # Judge for evaluation
        self.judge = JudgeAgent(base_model)
        
        # Planner for dynamic agent selection
        self.planner_model = base_model
        
    def discover(self, 
                problem_context: Dict[str, Any],
                target_data: Optional[Any] = None) -> ScientificDiscovery:
        """
        Main discovery loop implementing Xolver's approach.
        
        Args:
            problem_context: Problem specification and domain
            target_data: Target pattern/data to discover
            
        Returns:
            Best scientific discovery found
        """
        
        # Phase 1: Planning - Select agents for this problem
        selected_agents = self._plan_agent_team(problem_context)
        
        # Initialize shared memory
        self.shared_memory = []
        
        # Retrieve relevant episodic memories
        relevant_memories = self._retrieve_episodic_memories(problem_context)
        
        # Phase 2: Iterative Discovery
        for iteration in range(self.max_iterations):
            print(f"\n=== Iteration {iteration + 1} ===")
            
            # Each agent generates
            agent_outputs = []
            
            for agent in selected_agents:
                # Build context for agent
                agent_context = {
                    **problem_context,
                    'shared_memory': self.shared_memory,
                    'iteration': iteration
                }
                
                # Generate thought and response
                thought, response = agent.generate(
                    agent_context,
                    self.shared_memory,
                    relevant_memories
                )
                
                # Extract expression from response
                expression = self._extract_expression(response)
                
                agent_outputs.append((agent.role, thought, response, expression))
                
                print(f"\n{agent.role}:")
                print(f"Thought: {thought[:200]}...")
                print(f"Expression: {expression}")
            
            # Phase 3: Judge evaluates all outputs
            evaluations = self.judge.evaluate(agent_outputs, {
                'shared_memory': self.shared_memory,
                'target_data': target_data
            })
            
            # Update shared memory with top-k results
            self._update_shared_memory(evaluations)
            
            print(f"\nJudge Rankings:")
            for i, eval_result in enumerate(evaluations[:3]):
                print(f"{i+1}. {eval_result['agent_role']}: "
                      f"Score={eval_result['score']:.3f}, "
                      f"Feedback: {eval_result['feedback']}")
            
            # Check convergence
            if self._check_convergence():
                print("\nConverged!")
                break
        
        # Phase 4: Package best discovery
        best_result = self.shared_memory[0]
        discovery = ScientificDiscovery(
            expression=best_result['expression'],
            domain=problem_context.get('domain', 'unknown'),
            hypothesis=best_result.get('response', ''),
            evidence=[eval_result for eval_result in evaluations if eval_result['score'] > 0.8],
            confidence=best_result['score'],
            reasoning_trace=[eval_result['thought'] for eval_result in evaluations],
            agent_roles=[agent.role for agent in selected_agents],
            iteration_count=iteration + 1,
            validation_score=best_result.get('detailed_scores', {}).get('correctness', 0)
        )
        
        # Update episodic memory
        self._update_episodic_memory(discovery)
        
        return discovery
    
    def _plan_agent_team(self, problem_context: Dict[str, Any]) -> List[ScientificAgent]:
        """Dynamically select agents based on problem"""
        
        domain = problem_context.get('domain', 'general')
        
        # Always include hypothesis generator
        agents = [
            self.agent_types['hypothesis_generator'](self.planner_model, self.grammar)
        ]
        
        # Select additional agents based on domain
        if domain == 'attention_mechanisms':
            agents.extend([
                self.agent_types['symbolic_reasoner'](self.planner_model, self.grammar),
                self.agent_types['validator'](self.planner_model)
            ])
        elif domain == 'physics_laws':
            agents.extend([
                self.agent_types['experiment_designer'](self.planner_model),
                self.agent_types['validator'](self.planner_model)
            ])
        else:
            # Default team
            agents.extend([
                self.agent_types['symbolic_reasoner'](self.planner_model, self.grammar),
                self.agent_types['experiment_designer'](self.planner_model)
            ])
        
        return agents[:self.num_agents]
    
    def _retrieve_episodic_memories(self, 
                                   problem_context: Dict[str, Any],
                                   k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant discoveries from episodic memory"""
        
        if not self.episodic_memory:
            return []
        
        domain = problem_context.get('domain', '')
        
        # Filter by domain and sort by validation score
        relevant = [
            mem for mem in self.episodic_memory
            if mem.get('domain') == domain or domain == 'general'
        ]
        
        relevant.sort(key=lambda x: x.get('validation_score', 0), reverse=True)
        
        return relevant[:k]
    
    def _update_shared_memory(self, evaluations: List[Dict[str, Any]]):
        """Update shared memory with top results"""
        
        # Add all evaluations to candidates
        candidates = self.shared_memory + evaluations
        
        # Sort by score and keep top-k
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Remove duplicates based on expression
        seen_expressions = set()
        unique_candidates = []
        
        for candidate in candidates:
            expr = candidate.get('expression', '')
            if expr and expr not in seen_expressions:
                seen_expressions.add(expr)
                unique_candidates.append(candidate)
        
        self.shared_memory = unique_candidates[:self.num_agents]
    
    def _check_convergence(self) -> bool:
        """Check if discovery process has converged"""
        
        if not self.shared_memory:
            return False
        
        # Converged if top score is very high
        top_score = self.shared_memory[0].get('score', 0)
        if top_score > 0.95:
            return True
        
        # Or if top-k expressions are all similar
        if len(self.shared_memory) >= 3:
            top_expressions = [m['expression'] for m in self.shared_memory[:3]]
            if len(set(top_expressions)) == 1:
                return True
        
        return False
    
    def _extract_expression(self, response: str) -> str:
        """Extract symbolic expression from agent response"""
        
        # Look for patterns like "Expression: ..." or "Hypothesis: ..."
        import re
        
        patterns = [
            r'Expression:\s*([^\n]+)',
            r'Hypothesis:\s*([^\n]+)',
            r'```\s*([^`]+)\s*```'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1).strip()
        
        # Fallback: return first line that looks like an expression
        for line in response.split('\n'):
            if any(op in line for op in ['attention', 'softmax', 'linear', '+', '*', '(']):
                return line.strip()
        
        return "unknown_expression"
    
    def _update_episodic_memory(self, discovery: ScientificDiscovery):
        """Add successful discovery to long-term memory"""
        
        if discovery.validation_score > 0.8:  # Only store high-quality discoveries
            self.episodic_memory.append(discovery.to_episodic_entry())
            print(f"\nAdded discovery to episodic memory (total: {len(self.episodic_memory)})")


# Example usage
if __name__ == "__main__":
    from ...ml.networks.policy_networks import TransformerPolicy
    
    # Initialize components
    policy = TransformerPolicy(
        obs_dim=128,
        action_dim=20,
        hidden_dim=256,
        num_heads=8,
        num_layers=4
    )
    
    grammar = AIGrammar(
        primitives=['attention', 'linear', 'softmax', 'scale', 'mask'],
        max_depth=6
    )
    
    # Create Xolver system
    xolver_system = XolverScientificDiscoverySystem(
        base_model=policy,
        grammar=grammar,
        num_agents=3,
        max_iterations=3
    )
    
    # Run discovery
    problem_context = {
        'domain': 'attention_mechanisms',
        'task': 'discover_previous_token_attention',
        'description': 'Find symbolic expression for attention that focuses on previous token'
    }
    
    discovery = xolver_system.discover(problem_context)
    
    print("\n=== FINAL DISCOVERY ===")
    print(f"Expression: {discovery.expression}")
    print(f"Confidence: {discovery.confidence:.3f}")
    print(f"Validation Score: {discovery.validation_score:.3f}")
    print(f"Iterations: {discovery.iteration_count}")
    print(f"Agents involved: {', '.join(discovery.agent_roles)}")