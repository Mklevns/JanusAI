# janus/core/search/stats.py v6
"""
Statistics tracking and analysis for genetic algorithm search.

This module provides comprehensive statistics collection, analysis, and
reporting capabilities for genetic algorithm runs.
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Union, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import hashlib

def calculate_structural_fingerprint(expression: Expression) -> str:
    """
    Calculate a structural fingerprint of an expression tree.
    
    This ignores constants and variable names, focusing on the tree structure
    and operators. Useful for detecting expressions that differ only in constants.
    
    Args:
        expression: Expression to fingerprint
        
    Returns:
        Hash string representing the structural fingerprint
    """
    try:
        def _get_structure(expr: Expression) -> str:
            """Recursively get structural representation."""
            if expr.operator == 'var':
                return 'VAR'
            elif expr.operator == 'const':
                return 'CONST'
            else:
                # For operators, include the operator and recurse on operands
                if hasattr(expr, 'operands') and expr.operands:
                    operand_structures = []
                    for operand in expr.operands:
                        if isinstance(operand, Expression):
                            operand_structures.append(_get_structure(operand))
                        else:
                            # Handle non-Expression operands
                            if isinstance(operand, (int, float)):
                                operand_structures.append('CONST')
                            else:
                                operand_structures.append('VAR')
                    
                    # Sort operands for commutative operators to normalize structure
                    if expr.operator in ['+', '*', 'min', 'max']:
                        operand_structures.sort()
                    
                    return f"{expr.operator}({','.join(operand_structures)})"
                else:
                    return expr.operator
        
        structure = _get_structure(expression)
        return hashlib.md5(structure.encode()).hexdigest()[:16]
        
    except Exception:
        # If fingerprinting fails, return a default hash
        return "unknown"


def calculate_tree_edit_distance(expr1: Expression, expr2: Expression) -> int:
    """
    Calculate a simplified tree edit distance between two expressions.
    
    This is a lightweight approximation of tree edit distance that counts
    the number of nodes that differ between two expression trees.
    
    Args:
        expr1: First expression
        expr2: Second expression
        
    Returns:
        Edit distance (number of differing nodes)
    """
    try:
        def _get_nodes(expr: Expression) -> List[str]:
            """Get all nodes in expression tree."""
            nodes = [expr.operator]
            
            if hasattr(expr, 'operands') and expr.operands:
                for operand in expr.operands:
                    if isinstance(operand, Expression):
                        nodes.extend(_get_nodes(operand))
                    else:
                        # Handle constants and variables
                        if isinstance(operand, (int, float)):
                            nodes.append('const')
                        else:
                            nodes.append('var')
            
            return nodes
        
        nodes1 = _get_nodes(expr1)
        nodes2 = _get_nodes(expr2)
        
        # Simple edit distance: sum of lengths minus 2 * common elements
        set1 = set(nodes1)
        set2 = set(nodes2)
        common = len(set1.intersection(set2))
        
        return len(nodes1) + len(nodes2) - 2 * common
        
    except Exception:
        # If calculation fails, return maximum distance
        return 1000


def calculate_average_pairwise_distance(
    population: List[Expression], 
    sample_size: int = 20
) -> float:
    """
    Calculate average pairwise tree edit distance on a sample of the population.
    
    Args:
        population: Population of expressions
        sample_size: Number of pairs to sample for distance calculation
        
    Returns:
        Average pairwise distance
    """
    if len(population) < 2:
        return 0.0
    
    try:
        # Sample pairs to avoid O(n^2) computation
        distances = []
        max_samples = min(sample_size, (len(population) * (len(population) - 1)) // 2)
        
        sampled_pairs = set()
        attempts = 0
        while len(distances) < max_samples and attempts < max_samples * 3:
            i = np.random.randint(0, len(population))
            j = np.random.randint(0, len(population))
            
            if i != j and (i, j) not in sampled_pairs and (j, i) not in sampled_pairs:
                sampled_pairs.add((i, j))
                distance = calculate_tree_edit_distance(population[i], population[j])
                distances.append(distance)
            
            attempts += 1
        
        return np.mean(distances) if distances else 0.0
        
    except Exception:
        return 0.0


@dataclass
class GenerationStats:
    """Statistics for a single generation of the genetic algorithm."""
    
    generation: int
    timestamp: float
    
    # Fitness statistics
    best_fitness: float
    worst_fitness: float
    mean_fitness: float
    median_fitness: float
    std_fitness: float
    
    # Enhanced population diversity metrics
    diversity_structural: float  # String-based diversity (legacy)
    diversity_fitness: float
    diversity_fingerprint: float  # Structural fingerprint diversity
    diversity_tree_distance: float  # Average pairwise tree edit distance
    unique_expressions: int
    unique_fingerprints: int
    
    # Performance metrics
    evaluation_time: float
    generation_time: float
    evaluations_count: int
    
    # Convergence indicators
    fitness_improvement: float
    stagnation_count: int
    convergence_rate: float
    
    # Expression complexity
    mean_complexity: float
    min_complexity: int
    max_complexity: int
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class SearchStats:
    """Comprehensive statistics for an entire search run."""
    
    # Run configuration
    start_time: float
    end_time: Optional[float] = None
    total_generations: int = 0
    
    # Best solution tracking
    best_expression: Optional[Expression] = None
    best_fitness: float = -np.inf
    best_generation: int = -1
    
    # Performance tracking
    total_evaluations: int = 0
    total_evaluation_time: float = 0.0
    average_generation_time: float = 0.0
    
    # Convergence analysis
    convergence_generation: Optional[int] = None
    convergence_threshold: float = 1e-10
    final_diversity: float = 0.0
    
    # Generation-by-generation history
    generation_history: List[GenerationStats] = field(default_factory=list)
    
    # Error tracking
    evaluation_errors: int = 0
    generation_errors: int = 0
    
    def get_duration(self) -> float:
        """Get total search duration in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    def is_converged(self, improvement_threshold: float = 1e-10, 
                    stagnation_limit: int = 10) -> bool:
        """Check if search has converged."""
        if len(self.generation_history) < stagnation_limit:
            return False
        
        recent_improvements = [
            gen.fitness_improvement for gen in self.generation_history[-stagnation_limit:]
        ]
        
        return all(imp <= improvement_threshold for imp in recent_improvements)


class StatsTracker:
    """
    Tracks and analyzes statistics during genetic algorithm execution.
    
    This class provides comprehensive statistics collection and analysis
    capabilities for monitoring genetic algorithm performance and behavior.
    """
    
    def __init__(self, convergence_threshold: float = 1e-10):
        """
        Initialize statistics tracker.
        
        Args:
            convergence_threshold: Threshold for determining convergence
        """
        self.convergence_threshold = convergence_threshold
        self.search_stats = SearchStats(
            start_time=time.time(),
            convergence_threshold=convergence_threshold
        )
        
        # Internal tracking
        self._fitness_history = deque(maxlen=100)
        self._diversity_history = deque(maxlen=50)
        self._stagnation_counter = 0
        self._last_improvement = 0
        
        self.logger = logging.getLogger(__name__)
    
    def start_generation(self, generation: int) -> float:
        """Mark the start of a generation and return timestamp."""
        timestamp = time.time()
        self._generation_start_time = timestamp
        return timestamp
    
    def record_generation(
        self,
        generation: int,
        population: List[Expression],
        fitnesses: List[float],
        evaluation_time: float,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> GenerationStats:
        """
        Record statistics for a completed generation.
        
        Args:
            generation: Generation number
            population: Population of expressions
            fitnesses: Fitness values for population
            evaluation_time: Time spent on fitness evaluation
            additional_metadata: Additional metadata to store
            
        Returns:
            GenerationStats object for this generation
        """
        timestamp = time.time()
        generation_time = timestamp - self._generation_start_time
        
        # Calculate fitness statistics
        fitness_stats = self._calculate_fitness_stats(fitnesses)
        
        # Calculate diversity metrics
        diversity_metrics = self._calculate_diversity_metrics(population, fitnesses)
        
        # Calculate complexity statistics
        complexity_stats = self._calculate_complexity_stats(population)
        
        # Update convergence tracking
        convergence_metrics = self._update_convergence_tracking(
            fitness_stats['best'], generation
        )
        
        # Create generation stats
        gen_stats = GenerationStats(
            generation=generation,
            timestamp=timestamp,
            best_fitness=fitness_stats['best'],
            worst_fitness=fitness_stats['worst'],
            mean_fitness=fitness_stats['mean'],
            median_fitness=fitness_stats['median'],
            std_fitness=fitness_stats['std'],
            diversity_structural=diversity_metrics['structural'],
            diversity_fitness=diversity_metrics['fitness'],
            diversity_fingerprint=diversity_metrics['fingerprint'],
            diversity_tree_distance=diversity_metrics['tree_distance'],
            unique_expressions=diversity_metrics['unique_count'],
            unique_fingerprints=diversity_metrics['unique_fingerprints'],
            evaluation_time=evaluation_time,
            generation_time=generation_time,
            evaluations_count=len(fitnesses),
            fitness_improvement=convergence_metrics['improvement'],
            stagnation_count=convergence_metrics['stagnation'],
            convergence_rate=convergence_metrics['rate'],
            mean_complexity=complexity_stats['mean'],
            min_complexity=complexity_stats['min'],
            max_complexity=complexity_stats['max'],
            metadata=additional_metadata or {}
        )
        
        # Update search stats
        self._update_search_stats(gen_stats, population[np.argmax(fitnesses)])
        
        # Store generation stats
        self.search_stats.generation_history.append(gen_stats)
        
        return gen_stats
    
    def finish_search(self) -> SearchStats:
        """Mark the end of search and finalize statistics."""
        self.search_stats.end_time = time.time()
        self.search_stats.total_generations = len(self.search_stats.generation_history)
        
        if self.search_stats.generation_history:
            # Calculate average generation time
            total_gen_time = sum(g.generation_time for g in self.search_stats.generation_history)
            self.search_stats.average_generation_time = (
                total_gen_time / self.search_stats.total_generations
            )
            
            # Set final diversity
            self.search_stats.final_diversity = (
                self.search_stats.generation_history[-1].diversity_structural
            )
        
        return self.search_stats
    
    def get_convergence_analysis(self) -> Dict[str, Any]:
        """Get detailed convergence analysis."""
        if not self.search_stats.generation_history:
            return {"status": "no_data"}
        
        history = self.search_stats.generation_history
        
        # Fitness progression
        fitness_progression = [g.best_fitness for g in history]
        improvements = np.diff(fitness_progression)
        
        # Diversity progression
        diversity_progression = [g.diversity_structural for g in history]
        
        # Stagnation analysis
        recent_window = min(10, len(history))
        recent_improvements = improvements[-recent_window:] if len(improvements) >= recent_window else improvements
        
        analysis = {
            "status": "converged" if self.search_stats.is_converged() else "progressing",
            "total_generations": len(history),
            "best_fitness": self.search_stats.best_fitness,
            "best_generation": self.search_stats.best_generation,
            "convergence_generation": self.search_stats.convergence_generation,
            "final_diversity": self.search_stats.final_diversity,
            "fitness_progression": fitness_progression,
            "diversity_progression": diversity_progression,
            "recent_improvement_rate": np.mean(recent_improvements) if len(recent_improvements) > 0 else 0.0,
            "stagnation_period": self._stagnation_counter,
            "total_evaluations": self.search_stats.total_evaluations,
            "average_generation_time": self.search_stats.average_generation_time
        }
        
        return analysis
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.search_stats.generation_history:
            return {"status": "no_data"}
        
        history = self.search_stats.generation_history
        
        return {
            "total_duration": self.search_stats.get_duration(),
            "total_generations": len(history),
            "total_evaluations": self.search_stats.total_evaluations,
            "average_generation_time": self.search_stats.average_generation_time,
            "evaluations_per_second": (
                self.search_stats.total_evaluations / self.search_stats.get_duration()
                if self.search_stats.get_duration() > 0 else 0
            ),
            "evaluation_errors": self.search_stats.evaluation_errors,
            "generation_errors": self.search_stats.generation_errors,
            "final_best_fitness": self.search_stats.best_fitness,
            "final_diversity": self.search_stats.final_diversity
        }
    
    # Private helper methods
    
    def _calculate_fitness_stats(self, fitnesses: List[float]) -> Dict[str, float]:
        """Calculate comprehensive fitness statistics."""
        valid_fitnesses = [f for f in fitnesses if np.isfinite(f)]
        
        if not valid_fitnesses:
            return {
                'best': -np.inf, 'worst': -np.inf, 'mean': -np.inf,
                'median': -np.inf, 'std': 0.0
            }
        
        return {
            'best': max(valid_fitnesses),
            'worst': min(valid_fitnesses),
            'mean': np.mean(valid_fitnesses),
            'median': np.median(valid_fitnesses),
            'std': np.std(valid_fitnesses)
        }
    
    def _calculate_diversity_metrics(
        self, 
        population: List[Expression], 
        fitnesses: List[float]
    ) -> Dict[str, float]:
        """Calculate enhanced population diversity metrics."""
        
        # Structural diversity (unique expression strings) - legacy metric
        expression_strings = set()
        for expr in population:
            try:
                expr_str = expression_to_string(expr)
                expression_strings.add(expr_str)
            except Exception:
                pass  # Skip problematic expressions
        
        structural_diversity = len(expression_strings) / len(population) if population else 0.0
        
        # Structural fingerprint diversity (ignores constants)
        fingerprints = set()
        for expr in population:
            try:
                fingerprint = calculate_structural_fingerprint(expr)
                fingerprints.add(fingerprint)
            except Exception:
                pass
        
        fingerprint_diversity = len(fingerprints) / len(population) if population else 0.0
        
        # Average pairwise tree edit distance
        tree_distance = calculate_average_pairwise_distance(population, sample_size=20)
        
        # Fitness diversity (coefficient of variation)
        valid_fitnesses = [f for f in fitnesses if np.isfinite(f)]
        if len(valid_fitnesses) > 1:
            fitness_diversity = np.std(valid_fitnesses) / (abs(np.mean(valid_fitnesses)) + 1e-8)
        else:
            fitness_diversity = 0.0
        
        return {
            'structural': structural_diversity,
            'fitness': fitness_diversity,
            'fingerprint': fingerprint_diversity,
            'tree_distance': tree_distance,
            'unique_count': len(expression_strings),
            'unique_fingerprints': len(fingerprints)
        }
    
    def _calculate_complexity_stats(self, population: List[Expression]) -> Dict[str, Union[float, int]]:
        """Calculate expression complexity statistics."""
        complexities = []
        
        for expr in population:
            try:
                complexity = get_expression_complexity(expr)
                complexities.append(complexity)
            except Exception:
                complexities.append(1)  # Default complexity for problematic expressions
        
        if not complexities:
            return {'mean': 0.0, 'min': 0, 'max': 0}
        
        return {
            'mean': np.mean(complexities),
            'min': min(complexities),
            'max': max(complexities)
        }
    
    def _update_convergence_tracking(
        self, 
        current_best_fitness: float, 
        generation: int
    ) -> Dict[str, float]:
        """Update convergence tracking metrics."""
        
        # Calculate improvement
        if self._fitness_history:
            last_best = max(self._fitness_history)
            improvement = current_best_fitness - last_best
        else:
            improvement = 0.0
        
        # Update fitness history
        self._fitness_history.append(current_best_fitness)
        
        # Update stagnation counter
        if improvement <= self.convergence_threshold:
            self._stagnation_counter += 1
        else:
            self._stagnation_counter = 0
            self._last_improvement = generation
        
        # Calculate convergence rate (improvement per generation over recent history)
        if len(self._fitness_history) >= 5:
            recent_fitnesses = list(self._fitness_history)[-5:]
            convergence_rate = (recent_fitnesses[-1] - recent_fitnesses[0]) / 4
        else:
            convergence_rate = improvement
        
        return {
            'improvement': improvement,
            'stagnation': self._stagnation_counter,
            'rate': convergence_rate
        }
    
    def _update_search_stats(self, gen_stats: GenerationStats, best_expr: Expression):
        """Update overall search statistics."""
        
        # Update best solution tracking
        if gen_stats.best_fitness > self.search_stats.best_fitness:
            self.search_stats.best_fitness = gen_stats.best_fitness
            self.search_stats.best_expression = best_expr
            self.search_stats.best_generation = gen_stats.generation
        
        # Update performance counters
        self.search_stats.total_evaluations += gen_stats.evaluations_count
        self.search_stats.total_evaluation_time += gen_stats.evaluation_time
        
        # Check for convergence
        if (self.search_stats.convergence_generation is None and
            self.search_stats.is_converged()):
            self.search_stats.convergence_generation = gen_stats.generation


def create_stats_tracker(convergence_threshold: float = 1e-10) -> StatsTracker:
    """Factory function to create a statistics tracker."""
    return StatsTracker(convergence_threshold=convergence_threshold)


def analyze_search_run(stats: SearchStats) -> Dict[str, Any]:
    """
    Perform comprehensive analysis of a completed search run.
    
    Args:
        stats: SearchStats object from completed run
        
    Returns:
        Dictionary containing detailed analysis results
    """
    if not stats.generation_history:
        return {"error": "No generation data available"}
    
    history = stats.generation_history
    
    # Performance analysis
    performance = {
        "duration": stats.get_duration(),
        "generations": len(history),
        "evaluations": stats.total_evaluations,
        "avg_gen_time": stats.average_generation_time,
        "eval_rate": stats.total_evaluations / stats.get_duration() if stats.get_duration() > 0 else 0
    }
    
    # Solution quality analysis
    quality = {
        "best_fitness": stats.best_fitness,
        "best_generation": stats.best_generation,
        "final_diversity": stats.final_diversity,
        "convergence_generation": stats.convergence_generation
    }
    
    # Progression analysis
    fitness_values = [g.best_fitness for g in history]
    diversity_values = [g.diversity_structural for g in history]
    
    progression = {
        "fitness_trend": np.polyfit(range(len(fitness_values)), fitness_values, 1)[0],
        "diversity_trend": np.polyfit(range(len(diversity_values)), diversity_values, 1)[0],
        "early_improvement": fitness_values[min(10, len(fitness_values)-1)] - fitness_values[0] if len(fitness_values) > 1 else 0,
        "late_improvement": fitness_values[-1] - fitness_values[max(0, len(fitness_values)-11)] if len(fitness_values) > 10 else 0
    }
    
    return {
        "performance": performance,
        "quality": quality,
        "progression": progression,
        "summary": {
            "success": stats.best_fitness > -1e6,
            "converged": stats.is_converged(),
            "efficient": performance["eval_rate"] > 100,  # Arbitrary threshold
            "diverse": quality["final_diversity"] > 0.1
        }
    }