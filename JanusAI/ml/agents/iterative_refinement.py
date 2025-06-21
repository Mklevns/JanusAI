# JanusAI/agents/iterative_refinement.py
"""
Iterative refinement loop with convergence detection and feedback incorporation.
Integrates with existing Janus components (PPO, InterpretabilityReward, AIGrammar).
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from collections import deque

from janus_ai.ml.multi_agent_system import (

    BaseScientificAgent, DynamicAgentPool, PlannerAgent,
    AgentRole, AgentConfig
)
from janus_ai.memory.dual_memory_system import (

    DualMemorySystem, IntermediateResult, Discovery
)
from janus_ai.grammar.ai_grammar import AIGrammar

from janus_ai.rewards.interpretability_reward import InterpretabilityReward

from janus_ai.ml.training.advanced_ppo_trainer import AdvancedPPOTrainer, PPOConfig

from janus_ai.environments.symbolic_discovery_env import SymbolicDiscoveryEnv



@dataclass
class IterationMetrics:
    """Metrics for a single iteration"""
    iteration: int
    best_score: float
    average_score: float
    num_unique_expressions: int
    convergence_metric: float
    time_elapsed: float


class JudgeAgent:
    """Judge agent that evaluates and provides feedback"""
    
    def __init__(self,
                 reward_function: InterpretabilityReward,
                 grammar: AIGrammar):
        
        self.reward_function = reward_function
        self.grammar = grammar
        self.logger = logging.getLogger(__name__)
    
    def evaluate(self,
                result: IntermediateResult,
                problem_context: Dict[str, Any],
                shared_memory: List[IntermediateResult]) -> Tuple[float, Dict[str, float], str]:
        """
        Evaluate a result and provide detailed feedback.
        
        Returns:
            overall_score: Combined score
            detailed_scores: Breakdown of score components
            feedback: Natural language feedback
        """
        
        # Get expression tree representation
        try:
            expr_tree = self.grammar.parse_expression(result.expression)
        except:
            # Invalid expression
            return 0.0, {'validity': 0.0}, "Invalid expression syntax"
        
        # Calculate detailed scores
        detailed_scores = {}
        
        # 1. Validity and correctness
        detailed_scores['validity'] = 1.0 if expr_tree else 0.0
        
        # 2. Fidelity to target (if available)
        target_data = problem_context.get('target_data')
        if target_data is not None:
            fidelity = self.reward_function._calculate_fidelity(
                result.expression, target_data
            )
            detailed_scores['fidelity'] = fidelity
        else:
            detailed_scores['fidelity'] = 0.5  # No target, neutral score
        
        # 3. Simplicity
        simplicity = self.reward_function._calculate_simplicity(result.expression)
        detailed_scores['simplicity'] = simplicity
        
        # 4. Novelty compared to shared memory
        novelty = self._calculate_novelty(result.expression, shared_memory)
        detailed_scores['novelty'] = novelty
        
        # 5. Consistency
        consistency = self.reward_function._calculate_consistency(
            result.expression, problem_context
        )
        detailed_scores['consistency'] = consistency
        
        # 6. Theoretical soundness
        soundness = self._evaluate_theoretical_soundness(result.expression, problem_context)
        detailed_scores['theoretical_soundness'] = soundness
        
        # Calculate overall score with weights
        weights = {
            'validity': 0.2,
            'fidelity': 0.3,
            'simplicity': 0.15,
            'novelty': 0.15,
            'consistency': 0.1,
            'theoretical_soundness': 0.1
        }
        
        overall_score = sum(
            detailed_scores.get(key, 0) * weight 
            for key, weight in weights.items()
        )
        
        # Generate feedback
        feedback = self._generate_feedback(result, detailed_scores, shared_memory)
        
        return overall_score, detailed_scores, feedback
    
    def _calculate_novelty(self, expression: str, shared_memory: List[IntermediateResult]) -> float:
        """Calculate novelty score compared to existing attempts"""
        
        if not shared_memory:
            return 1.0
        
        existing_expressions = [r.expression for r in shared_memory]
        
        # Exact match check
        if expression in existing_expressions:
            return 0.1
        
        # Structural similarity check
        max_similarity = 0.0
        for existing in existing_expressions:
            similarity = self._expression_similarity(expression, existing)
            max_similarity = max(max_similarity, similarity)
        
        return 1.0 - max_similarity
    
    def _expression_similarity(self, expr1: str, expr2: str) -> float:
        """Calculate similarity between expressions"""
        # Simple token-based similarity
        tokens1 = set(expr1.replace('(', ' ').replace(')', ' ').split())
        tokens2 = set(expr2.replace('(', ' ').replace(')', ' ').split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union
    
    def _evaluate_theoretical_soundness(self, expression: str, context: Dict[str, Any]) -> float:
        """Evaluate theoretical soundness of expression"""
        
        score = 1.0
        
        # Domain-specific checks
        if context.get('domain') == 'attention_mechanisms':
            # Should have Query and Key operations
            if 'Q' not in expression or 'K' not in expression:
                score *= 0.5
            
            # Should have normalization
            if 'softmax' not in expression and 'normalize' not in expression:
                score *= 0.7
            
            # Should have scaling for stability
            if 'sqrt' not in expression and 'scale' not in expression:
                score *= 0.8
        
        elif context.get('domain') == 'physics_laws':
            # Should be dimensionally consistent
            # (simplified check)
            if expression.count('*') != expression.count('/'):
                score *= 0.8
        
        return score
    
    def _generate_feedback(self, 
                          result: IntermediateResult,
                          scores: Dict[str, float],
                          shared_memory: List[IntermediateResult]) -> str:
        """Generate constructive feedback"""
        
        feedback_parts = []
        
        # Identify strengths
        strengths = [k for k, v in scores.items() if v > 0.7]
        if strengths:
            feedback_parts.append(f"Strengths: {', '.join(strengths)}")
        
        # Identify weaknesses
        weaknesses = [(k, v) for k, v in scores.items() if v < 0.5]
        for weakness, score in weaknesses:
            if weakness == 'validity':
                feedback_parts.append("Issue: Invalid expression syntax")
            elif weakness == 'fidelity':
                feedback_parts.append("Issue: Poor match to target pattern")
            elif weakness == 'simplicity':
                feedback_parts.append("Issue: Expression is too complex")
            elif weakness == 'novelty':
                feedback_parts.append("Issue: Too similar to previous attempts")
            elif weakness == 'theoretical_soundness':
                feedback_parts.append("Issue: Violates theoretical principles")
        
        # Suggest improvements
        if scores.get('simplicity', 1.0) < 0.5:
            feedback_parts.append("Suggestion: Simplify by removing redundant operations")
        
        if scores.get('theoretical_soundness', 1.0) < 0.7:
            if 'softmax' not in result.expression:
                feedback_parts.append("Suggestion: Add softmax for probability normalization")
            if 'sqrt' not in result.expression:
                feedback_parts.append("Suggestion: Add scaling factor sqrt(d) for stability")
        
        return "; ".join(feedback_parts)


class IterativeRefinementLoop:
    """
    Main iterative refinement loop that orchestrates the discovery process.
    Integrates all components and manages convergence.
    """
    
    def __init__(self,
                 memory_system: DualMemorySystem,
                 grammar: AIGrammar,
                 reward_function: InterpretabilityReward,
                 ppo_trainer: Optional[AdvancedPPOTrainer] = None,
                 convergence_patience: int = 3):
        
        self.memory_system = memory_system
        self.grammar = grammar
        self.reward_function = reward_function
        self.ppo_trainer = ppo_trainer
        self.convergence_patience = convergence_patience
        
        # Create judge
        self.judge = JudgeAgent(reward_function, grammar)
        
        # Metrics tracking
        self.iteration_history: List[IterationMetrics] = []
        self.convergence_scores = deque(maxlen=convergence_patience)
        
        self.logger = logging.getLogger(__name__)
    
    def run_discovery(self,
                     problem_context: Dict[str, Any],
                     agent_pool: DynamicAgentPool,
                     planner: PlannerAgent,
                     max_iterations: int = 10) -> Discovery:
        """
        Run complete discovery process with iterative refinement.
        
        Args:
            problem_context: Problem specification
            agent_pool: Pool of available agents
            planner: Planner for team composition
            max_iterations: Maximum iterations
            
        Returns:
            Final discovery
        """
        
        start_time = datetime.now()
        problem_id = f"discovery_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Starting discovery process for problem {problem_id}")
        
        # Initialize shared memory
        shared_memory = self.memory_system.start_problem(problem_id)
        
        # Get relevant episodic memories
        episodic_memories = self._retrieve_relevant_memories(problem_context)
        
        # Plan initial team
        team = planner.plan_discovery_team(problem_context)
        
        # Main iteration loop
        iteration = 0
        converged = False
        
        while iteration < max_iterations and not converged:
            iteration_start = datetime.now()
            
            self.logger.info(f"\n=== Iteration {iteration + 1} ===")
            
            # Phase 1: Agent generation
            agent_results = self._run_agent_generation(
                team, problem_id, problem_context, 
                shared_memory.get_top(10), episodic_memories
            )
            
            # Phase 2: Evaluation and feedback
            evaluated_results = self._evaluate_results(
                agent_results, problem_context, shared_memory.get_top(10)
            )
            
            # Phase 3: Update shared memory
            for result, (score, detailed_scores, feedback) in evaluated_results:
                result.score = score
                result.detailed_scores = detailed_scores
                result.feedback = feedback
                shared_memory.add(result)
                
                # Update agent performance
                agent_pool.update_performance(result.agent_role, score)
            
            # Phase 4: Training update (if PPO trainer available)
            if self.ppo_trainer and iteration % 2 == 0:
                self._run_training_update(shared_memory.get_top(5))
            
            # Phase 5: Calculate metrics
            metrics = self._calculate_iteration_metrics(
                iteration, shared_memory, iteration_start
            )
            self.iteration_history.append(metrics)
            
            # Phase 6: Check convergence
            converged = self._check_convergence(metrics, shared_memory)
            
            # Phase 7: Adapt team if needed
            if not converged and iteration % 3 == 0:
                performance_data = {
                    'metrics': metrics,
                    'agent_scores': {
                        agent.agent_id: agent_pool._get_average_score(agent.agent_id)
                        for agent in team
                    }
                }
                team = planner.adapt_team(team, performance_data)
            
            # Log progress
            self._log_iteration_progress(metrics, shared_memory)
            
            iteration += 1
        
        # Create final discovery
        discovery = self._create_final_discovery(
            problem_id, problem_context, shared_memory
        )
        
        # Save to episodic memory
        if discovery.validation_score > 0.7:  # Quality threshold
            self.memory_system.episodic.add(discovery)
        
        self.logger.info(f"Discovery complete: {discovery.expression} (score: {discovery.validation_score:.3f})")
        
        return discovery
    
    def _run_agent_generation(self,
                            team: List[BaseScientificAgent],
                            problem_id: str,
                            problem_context: Dict[str, Any],
                            shared_memory: List[IntermediateResult],
                            episodic_memories: List[Discovery]) -> List[IntermediateResult]:
        """Run generation phase with all agents"""
        
        results = []
        
        for agent in team:
            try:
                result = agent.act(
                    problem_id, problem_context, 
                    shared_memory, episodic_memories
                )
                results.append(result)
                
                self.logger.debug(f"{agent.role.value}: {result.expression}")
                
            except Exception as e:
                self.logger.error(f"Agent {agent.agent_id} failed: {e}")
        
        return results
    
    def _evaluate_results(self,
                         results: List[IntermediateResult],
                         problem_context: Dict[str, Any],
                         shared_memory: List[IntermediateResult]) -> List[Tuple]:
        """Evaluate all results with judge"""
        
        evaluated = []
        
        for result in results:
            score, detailed_scores, feedback = self.judge.evaluate(
                result, problem_context, shared_memory
            )
            
            evaluated.append((result, (score, detailed_scores, feedback)))
            
            # Update agent with feedback
            agent_id = result.agent_role  # Would need actual agent reference
            # agent.update_from_feedback(feedback, score)
        
        # Sort by score
        evaluated.sort(key=lambda x: x[1][0], reverse=True)
        
        return evaluated
    
    def _run_training_update(self, top_results: List[IntermediateResult]):
        """Update policy networks based on results"""
        
        if not self.ppo_trainer or not top_results:
            return
        
        # Convert results to training data
        # This is simplified - actual implementation would depend on your setup
        
        observations = []  # Would extract from problem context
        actions = []      # Would extract from expressions
        rewards = [r.score for r in top_results]
        
        # Run PPO update
        # self.ppo_trainer.update(observations, actions, rewards)
        
        self.logger.debug("Completed training update")
    
    def _calculate_iteration_metrics(self,
                                   iteration: int,
                                   shared_memory,
                                   start_time: datetime) -> IterationMetrics:
        """Calculate metrics for current iteration"""
        
        all_results = shared_memory.results
        
        if all_results:
            scores = [r.score for r in all_results]
            best_score = max(scores)
            average_score = np.mean(scores)
            unique_expressions = len(shared_memory.get_unique_expressions())
            
            # Convergence metric: variance of top scores
            top_scores = [r.score for r in all_results[:3]]
            convergence_metric = 1.0 - np.var(top_scores) if len(top_scores) > 1 else 0.0
        else:
            best_score = 0.0
            average_score = 0.0
            unique_expressions = 0
            convergence_metric = 0.0
        
        time_elapsed = (datetime.now() - start_time).total_seconds()
        
        return IterationMetrics(
            iteration=iteration,
            best_score=best_score,
            average_score=average_score,
            num_unique_expressions=unique_expressions,
            convergence_metric=convergence_metric,
            time_elapsed=time_elapsed
        )
    
    def _check_convergence(self, 
                          metrics: IterationMetrics,
                          shared_memory) -> bool:
        """Check if discovery has converged"""
        
        # Update convergence history
        self.convergence_scores.append(metrics.best_score)
        
        # Perfect score
        if metrics.best_score > 0.95:
            self.logger.info("Converged: Found near-perfect solution")
            return True
        
        # Score plateau
        if len(self.convergence_scores) == self.convergence_patience:
            score_variance = np.var(list(self.convergence_scores))
            if score_variance < 0.001:
                self.logger.info("Converged: Scores have plateaued")
                return True
        
        # No improvement for patience iterations
        if len(self.iteration_history) > self.convergence_patience:
            recent_best = max(m.best_score for m in self.iteration_history[-self.convergence_patience:])
            older_best = max(m.best_score for m in self.iteration_history[:-self.convergence_patience])
            
            if recent_best <= older_best:
                self.logger.info("Converged: No improvement in recent iterations")
                return True
        
        return False
    
    def _retrieve_relevant_memories(self, 
                                  problem_context: Dict[str, Any]) -> List[Discovery]:
        """Retrieve relevant discoveries from episodic memory"""
        
        domain = problem_context.get('domain')
        
        # Get top discoveries in domain
        if domain:
            memories = self.memory_system.episodic.search_by_domain(domain, limit=5)
        else:
            memories = self.memory_system.episodic.get_top_validated(5)
        
        self.logger.info(f"Retrieved {len(memories)} relevant memories")
        
        return memories
    
    def _create_final_discovery(self,
                              problem_id: str,
                              problem_context: Dict[str, Any],
                              shared_memory) -> Discovery:
        """Create final discovery from best result"""
        
        # Get best result
        top_results = shared_memory.get_top(1)
        
        if not top_results:
            # No valid results
            return Discovery(
                id=problem_id,
                timestamp=datetime.now(),
                domain=problem_context.get('domain', 'unknown'),
                expression="no_valid_discovery",
                hypothesis="Failed to find valid expression",
                evidence=[],
                confidence=0.0,
                validation_score=0.0,
                reasoning_trace=["No valid results produced"],
                agent_roles=[]
            )
        
        best_result = top_results[0]
        
        # Aggregate evidence
