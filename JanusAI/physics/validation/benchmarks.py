"""
Physics Benchmarks
==================

Defines standard benchmark tasks and evaluation suites for assessing the
performance of the physics discovery system. These benchmarks leverage
known physical laws and data generators to provide quantitative metrics.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Callable

# Import necessary components from other modules
from janus_ai.physics.data.task_distribution import PhysicsTask, PhysicsTaskDistribution
from janus_ai.physics.validation.known_laws import KnownLawLibrary
from janus_ai.core.expressions.expression import Expression, Variable
from janus_ai.core.expressions.symbolic_math import evaluate_expression_on_data, are_expressions_equivalent_sympy


class BenchmarkSuite:
    """
    A suite of predefined benchmark tasks for evaluating physics discovery.
    """
    def __init__(self, task_distribution: PhysicsTaskDistribution, known_laws: KnownLawLibrary):
        """
        Initializes the BenchmarkSuite.

        Args:
            task_distribution: An instance of PhysicsTaskDistribution to get benchmark tasks.
            known_laws: An instance of KnownLawLibrary for ground truth law validation.
        """
        self.task_distribution = task_distribution
        self.known_laws = known_laws
        self._benchmark_tasks: List[PhysicsTask] = self._define_benchmark_tasks()

    def _define_benchmark_tasks(self) -> List[PhysicsTask]:
        """
        Defines the specific tasks that constitute this benchmark suite.
        These are typically a subset of tasks from the general distribution,
        chosen for their representativeness or challenge.
        """
        benchmark_task_names = [
            "harmonic_oscillator_energy",
            "pendulum_small_angle",
            "kepler_orbit",
            "ideal_gas_law",
            "coulomb_law",
            "elastic_collision",
            "double_pendulum_energy" # A more challenging one, good for advanced benchmarks
        ]
        
        defined_tasks = []
        for name in benchmark_task_names:
            try:
                task = self.task_distribution.get_task_by_name(name)
                defined_tasks.append(task)
            except ValueError:
                print(f"Warning: Benchmark task '{name}' not found in task distribution library. Skipping.")
        return defined_tasks

    def run_benchmark(self,
                      agent_expression_generator: Callable[[PhysicsTask, int], Union[str, Expression]],
                      n_eval_samples_per_task: int = 1000,
                      n_task_instances_per_benchmark: int = 5 # Number of times to run each task with new data
                     ) -> Dict[str, Any]:
        """
        Runs the benchmark suite, evaluating an agent's ability to discover laws.

        Args:
            agent_expression_generator: A callable that takes a `PhysicsTask` and a `seed`
                                        and returns the agent's discovered expression (string or Expression object).
                                        This simulates the agent's output for a given task.
            n_eval_samples_per_task: Number of data points to generate for evaluating each discovered expression.
            n_task_instances_per_benchmark: How many times to generate fresh data and re-evaluate for each benchmark task.

        Returns:
            A dictionary containing aggregated benchmark results, including:
            - 'task_results': List of results for each task run.
            - 'average_fidelity': Overall average fidelity score across all tasks.
            - 'correct_discovery_rate': Percentage of tasks where the correct law was discovered.
        """
        all_task_results = []
        
        print(f"\n--- Running Benchmark Suite ({len(self._benchmark_tasks)} tasks) ---")
        print(f"  Evaluating {n_task_instances_per_benchmark} instances per task.")

        for task_idx, task in enumerate(self._benchmark_tasks):
            print(f"\nEvaluating Task {task_idx+1}/{len(self._benchmark_tasks)}: '{task.name}' (Difficulty: {task.difficulty:.1f})")
            
            task_fidelity_scores = []
            task_correct_discoveries = 0

            for instance_idx in range(n_task_instances_per_benchmark):
                # Generate fresh data for this instance of the task
                task_data_for_eval = task.generate_data(n_eval_samples_per_task, add_noise=True)
                
                # Simulate agent's discovery for this task instance
                try:
                    # Provide a unique seed for the agent's generation for diversity
                    discovered_expr = agent_expression_generator(task, instance_idx) 
                    if discovered_expr is None:
                        print(f"    Instance {instance_idx+1}: Agent failed to generate expression. Skipping.")
                        continue
                except Exception as e:
                    print(f"    Instance {instance_idx+1}: Agent expression generation failed: {e}. Skipping.")
                    continue

                # Get the true law from the library
                true_law_obj = self.known_laws.get_law_by_name(task.name) # Assume task.name matches known_law name
                if true_law_obj is None:
                    print(f"    Warning: True law for task '{task.name}' not found in KnownLawLibrary. Cannot validate equivalence.")
                    true_law_obj = KnownLaw(name=task.name, formula_str=task.true_law, variables=task.variables, domain=task.domain) # Create ad-hoc for fidelity
                
                # Evaluate fidelity (how well the discovered expression fits the generated data)
                fidelity_score = self._calculate_expression_fidelity(
                    discovered_expr, 
                    task_data_for_eval, 
                    task.variables # Pass variable names
                )
                task_fidelity_scores.append(fidelity_score)

                # Check for equivalence to the true law
                is_correct = False
                if true_law_obj: # If a true law was found/created
                    try:
                        # Convert variables to SymPy Symbols for equivalence check
                        var_symbols = [sp.Symbol(v) for v in task.variables] # Assume task.variables are strings
                        is_correct = true_law_obj.is_equivalent(discovered_expr, tolerance=1e-3)
                        if is_correct:
                            task_correct_discoveries += 1
                    except Exception as e:
                        # print(f"    Equivalence check failed for {discovered_expr}: {e}")
                        pass # is_correct remains False

                print(f"    Instance {instance_idx+1}: Fidelity={fidelity_score:.4f}, Correct_Law={is_correct}")

            avg_fidelity = np.mean(task_fidelity_scores) if task_fidelity_scores else 0.0
            correct_rate = task_correct_discoveries / n_task_instances_per_benchmark if n_task_instances_per_benchmark > 0 else 0.0

            all_task_results.append({
                'task_name': task.name,
                'difficulty': task.difficulty,
                'avg_fidelity': avg_fidelity,
                'correct_discovery_rate': correct_rate,
                'total_instances': n_task_instances_per_benchmark
            })
            print(f"  Task '{task.name}' Summary: Avg Fidelity={avg_fidelity:.4f}, Correct Rate={correct_rate:.2f}")

        # Aggregate overall results
        overall_avg_fidelity = np.mean([res['avg_fidelity'] for res in all_task_results]) if all_task_results else 0.0
        overall_correct_discovery_rate = np.mean([res['correct_discovery_rate'] for res in all_task_results]) if all_task_results else 0.0

        print("\n--- Benchmark Suite Summary ---")
        print(f"Overall Average Fidelity: {overall_avg_fidelity:.4f}")
        print(f"Overall Correct Discovery Rate: {overall_correct_discovery_rate:.2f}")

        return {
            'task_results': all_task_results,
            'overall_average_fidelity': overall_avg_fidelity,
            'overall_correct_discovery_rate': overall_correct_discovery_rate,
            'benchmark_tasks_count': len(self._benchmark_tasks)
        }

    def _calculate_expression_fidelity(self, 
                                       expression: Union[str, Expression], 
                                       data: np.ndarray, 
                                       variable_names: List[str] # List of variable names in data
                                      ) -> float:
        """
        Calculates how well a discovered expression fits a given dataset (fidelity).
        Uses R-squared. This is a simplified version of FidelityReward.calculate_reward.
        """
        
        # Prepare data_dict for evaluate_expression_on_data
        # Assumes `data` columns map directly to `variable_names`
        evaluation_data_dict = {}
        for idx, var_name in enumerate(variable_names):
            if idx < data.shape[1]: # Ensure index is valid
                evaluation_data_dict[var_name] = data[:, idx]
        
        # Assume the last column of data is the true target output (y_true)
        # This convention needs to be consistent with how data is generated.
        if data.shape[1] < len(variable_names):
            print(f"Warning: Data columns ({data.shape[1]}) fewer than variable names ({len(variable_names)}). Assuming target is the last listed variable.")
            target_output_idx = -1 # Default to last column
            y_true = data[:, target_output_idx]
        else:
            # If data has exactly as many columns as variables, and the last variable is the target.
            # Otherwise, you need a way to specify which column is the target.
            # For simplicity, if variable_names match data columns, the *last* variable in the list
            # is assumed to be the target variable to be predicted.
            target_var_name = variable_names[-1]
            y_true = data[:, variable_names.index(target_var_name)]


        try:
            predicted_values = evaluate_expression_on_data(str(expression), evaluation_data_dict)
            
            if predicted_values is None or predicted_values.size == 0:
                return 0.0

            # Ensure valid, finite numbers for comparison
            predicted_values = np.asarray(predicted_values).flatten()
            y_true = np.asarray(y_true).flatten()

            valid_mask = np.isfinite(predicted_values) & np.isfinite(y_true)
            if not np.any(valid_mask):
                return 0.0

            pred_valid = predicted_values[valid_mask]
            true_valid = y_true[valid_mask]

            if len(pred_valid) == 0:
                return 0.0

            # Calculate R-squared
            ss_res = np.sum((true_valid - pred_valid) ** 2)
            ss_tot = np.sum((true_valid - np.mean(true_valid)) ** 2)
            
            if ss_tot < 1e-10: # True data is constant
                return 1.0 if ss_res < 1e-10 else 0.0 # Perfect if prediction is also constant and matches
            
            r_squared = 1 - (ss_res / ss_tot)
            
            return max(0.0, r_squared) # Clip R-squared to be non-negative
        except Exception as e:
            # print(f"Fidelity calculation failed for expression '{expression}': {e}")
            return 0.0 # Return 0 for calculation errors


if __name__ == "__main__":
    # Mock evaluate_expression_on_data and are_expressions_equivalent_sympy
    # if `janus.core.expressions.symbolic_math` is not ready.
    try:
        pass
        # from janus.core.expressions.symbolic_math import evaluate_expression_on_data as real_eval_expr_on_data
        # from janus.core.expressions.symbolic_math import are_expressions_equivalent_sympy as real_are_eq_sympy
    except ImportError:
        print("Using mock symbolic_math utilities for benchmarks.py test.")
        def evaluate_expression_on_data(expr_str: str, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
            # Simple mock evaluation: assumes 'x' as input, returns 3*x
            x_data = data_dict.get('x_0', data_dict.get('x', np.linspace(0, 1, 100)))
            if 'x' in expr_str or 'x_0' in expr_str:
                return x_data * 3.0 + np.random.rand(x_data.shape[0]) * 0.05 # Add some noise
            return np.full_like(x_data, 5.0) # Default constant

        def are_expressions_equivalent_sympy(expr1: sp.Expr, expr2: sp.Expr, symbols: List[sp.Symbol], tolerance: float) -> bool:
            # Very basic mock for equivalence: True if formulas match string, or simple forms
            str_expr1 = str(expr1.simplify())
            str_expr2 = str(expr2.simplify())
            return (str_expr1 == str_expr2) or (str_expr1 == "a*m" and str_expr2 == "m*a") # Example

    # Mock Expression and Variable classes if needed for `KnownLaw.is_equivalent` tests
    try:
        from janus.core.expressions.expression import Expression as RealExpression, Variable as RealVariable
    except ImportError:
        print("Using mock Expression and Variable for benchmarks.py test.")
        @dataclass(eq=True, frozen=False)
        class RealVariable:
            name: str
            index: int
            properties: Dict[str, Any] = field(default_factory=dict)
            symbolic: sp.Symbol = field(init=False)
            def __post_init__(self): self.symbolic = sp.Symbol(self.name)
            def __hash__(self): return hash((self.name, self.index))
            def __str__(self): return self.name
        @dataclass(eq=False, frozen=False)
        class RealExpression:
            operator: str
            operands: List[Any]
            _symbolic: Optional[sp.Expr] = field(init=False, repr=False)
            def __post_init__(self):
                if self.operator == 'var' and isinstance(self.operands[0], RealVariable):
                    self._symbolic = self.operands[0].symbolic
                elif self.operator == 'const':
                    self._symbolic = sp.Float(self.operands[0])
                elif self.operator == '+':
                    self._symbolic = self.operands[0].symbolic + self.operands[1].symbolic if all(hasattr(o, 'symbolic') for o in self.operands) else sp.Symbol('dummy_add')
                else: self._symbolic = sp.sympify(self.operator + "(" + ",".join([str(op) for op in self.operands]) + ")")
            @property
            def symbolic(self) -> sp.Expr: return self._symbolic
            def __str__(self) -> str: return str(self.symbolic)
    Expression = RealExpression
    Variable = RealVariable


    # --- Setup Benchmark Suite ---
    print("--- Testing BenchmarkSuite ---")

    # Mock PhysicsTaskDistribution and KnownLawLibrary
    class MockPhysicsTaskDistribution:
        def get_task_by_name(self, name: str) -> PhysicsTask:
            # Return a dummy PhysicsTask for testing. Assume minimal data generation.
            if name == "harmonic_oscillator_energy":
                # Data for [x, v, k, m, E]
                def gen_data_ho(n): return np.random.rand(n, 5) * 10
                return PhysicsTask(name, gen_data_ho, "0.5 * m * v**2 + 0.5 * k * x**2", ["x", "v", "k", "m", "E"], {}, [], ["energy"], 0.2, "mechanics")
            elif name == "Newton's Second Law": # Example for force discovery
                def gen_data_newton(n): return np.random.rand(n, 3) * 5 # m, a, F
                return PhysicsTask(name, gen_data_newton, "m*a", ["m", "a", "F"], {}, [], [], 0.1, "mechanics")
            elif name == "kepler_orbit":
                 def gen_data_kepler(n): return np.random.rand(n, 5) * 100 # r, v, G, M, F_grav
                 return PhysicsTask(name, gen_data_kepler, "G*M/r**2", ["r", "v", "G", "M", "F_grav"], {}, [], [], 0.6, "mechanics")
            elif name == "double_pendulum_energy":
                def gen_data_dp(n): return np.random.rand(n, 10) * 50 # Highly complex, just for structure
                return PhysicsTask(name, gen_data_dp, "complex_dp_energy_formula", ["t1", "t2", "o1", "o2", "m1", "m2", "L1", "L2", "g", "E"], {}, [], ["energy"], 0.9, "mechanics")
            # Add other benchmark tasks as mocks here...
            raise ValueError(f"Mock task '{name}' not found.")
        def describe_task_distribution(self): return "Mock Task Distribution"

    class MockKnownLawLibrary:
        def get_law_by_name(self, name: str) -> Optional[KnownLaw]:
            if name == "Newton's Second Law":
                return KnownLaw(name, "m*a", ["m", "a"], "mechanics")
            elif name == "Harmonic Oscillator Energy":
                return KnownLaw(name, "0.5 * m * v**2 + 0.5 * k * x**2", ["m", "v", "k", "x"], "mechanics")
            elif name == "kepler_orbit": # True law for Kepler
                return KnownLaw(name, "G*M/r**2", ["G", "M", "r"], "mechanics")
            elif name == "double_pendulum_energy":
                # For DP, the true law is very long, so we only compare it to itself
                long_dp_formula = "m1*g*L1*(1-cos(theta1)) + m2*g*(L1*(1-cos(theta1)) + L2*(1-cos(theta2))) + 0.5*m1*(L1*omega1)**2 + 0.5*m2*(L1**2*omega1**2 + L2**2*omega2**2 + 2*L1*L2*omega1*omega2*cos(theta1-theta2))"
                return KnownLaw(name, long_dp_formula, ["theta1", "theta2", "omega1", "omega2", "m1", "m2", "L1", "L2", "g"], "mechanics")
            return None
        def get_all_laws(self): return []

    task_dist_mock = MockPhysicsTaskDistribution()
    known_laws_mock = MockKnownLawLibrary()

    benchmark_suite = BenchmarkSuite(task_dist_mock, known_laws_mock)

    # --- Dummy Agent Expression Generator ---
    def dummy_agent_generator(task: PhysicsTask, seed: int) -> Union[str, Expression]:
        # This agent is 'smart' for HO energy, 'okay' for Newton, 'bad' for others.
        if task.name == "harmonic_oscillator_energy":
            return "0.5 * m * v**2 + 0.5 * k * x**2" # Correct form
        elif task.name == "Newton's Second Law":
            return "m*a" # Correct form
        elif task.name == "kepler_orbit":
            return "G * M / r**2" # Correct form
        elif task.name == "double_pendulum_energy":
            # For this complex task, return something 'plausible' but maybe not perfect
            return "m1*g*L1*(1-cos(theta1)) + m2*g*L2*(1-cos(theta2))" # Only potential energy part
        else:
            return "x + v" # Incorrect for most others

    results = benchmark_suite.run_benchmark(dummy_agent_generator, n_task_instances_per_benchmark=2)

    print("\nFinal Benchmark Results (from return value):\n", results)
    assert results['overall_correct_discovery_rate'] > 0.0 # Should be > 0 due to HO and Newton
    assert results['overall_average_fidelity'] > 0.0 # Should be > 0

