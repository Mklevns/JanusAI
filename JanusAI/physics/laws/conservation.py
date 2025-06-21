"""
Conservation Laws (ConservationBiasedReward as a Detector)
=========================================================

This module defines the `ConservationBiasedReward` class, which, in this context,
acts as a detector and quantifier of adherence to physical conservation laws
(e.g., energy, momentum) within trajectories. It includes mechanisms for
calculating violations and assessing entropy production, serving as a utility
for evaluating how well a discovered hypothesis conserves physical quantities.

NOTE: While named "Reward", its primary function here is to *detect* and *quantify*
conservation adherence for use by other reward systems (e.g., intrinsic rewards),
rather than being a `BaseReward` subclass itself.
"""

import numpy as np
import sympy as sp # For symbolic evaluation utilities
from typing import List, Dict, Any, Optional, Union
import time # For timestamp in history

# Import Expression and Variable for symbolic evaluation, from their new location
from janus_ai.core.expressions.expression import Expression, Variable
# Import evaluation utility from symbolic_math
from janus_ai.core.expressions.symbolic_math import evaluate_expression_on_data


class ConservationBiasedReward: # Retaining original class name as per instruction
    """
    Calculates a reward bonus based on the adherence to specified conservation laws.

    This class is designed to be used as a *detector* and *quantifier* for how well
    an agent's actions or discovered hypotheses conserve physical quantities
    like energy, momentum, etc. It also includes a mechanism to penalize
    entropy production, approximated by the irreversibility of energy changes
    in forward and backward trajectories.

    Attributes:
        conservation_types (List[str]): A list of conservation laws to check (e.g., ['energy', 'momentum']).
        weight_factor (float): A factor to scale the computed conservation bonus.
                               This is used when this detector's output is integrated into a larger reward.
        tolerances (Dict[str, float]): A dictionary mapping conservation types to their
                                       allowed violation tolerances. Violations within
                                       tolerance receive a higher "bonus" score.
        history (Dict[str, List[Dict[str, Any]]]): Stores a history of violations and bonuses
                                                     for each conservation type, useful for diagnostics.
    """
    def __init__(self, conservation_types: List[str], weight_factor: float) -> None:
        """
        Initializes the ConservationBiasedReward instance.

        Args:
            conservation_types: A list of strings identifying the conservation laws
                                to be evaluated (e.g., "energy", "momentum").
            weight_factor: A float that scales the output "bonus" value.
        """
        self.conservation_types: List[str] = conservation_types
        self.weight_factor: float = weight_factor
        self.tolerances: Dict[str, float] = {
            'energy': 1e-3,
            'momentum': 1e-4,
            'mass': 1e-5,
            'angular_momentum': 1e-4
        }
        # History stores dictionaries containing floats for 'violation', 'bonus', and Any for 'timestamp'
        self.history: Dict[str, List[Dict[str, Any]]] = {}

    def _calculate_violation(self,
                             predicted_val: Optional[Union[np.ndarray, float, List[float]]],
                             ground_truth_val: Optional[Union[np.ndarray, float, List[float]]],
                             c_type: str) -> float:
        """
        Calculates the normalized violation between predicted and ground truth conserved quantities.

        The violation is normalized by the magnitude of the ground truth value to make it
        relative. Returns 1.0 (max violation) if input data is missing or shapes mismatch.
        """
        if predicted_val is None or ground_truth_val is None:
            return 1.0  # Max violation if data is missing

        try:
            predicted_np = np.asarray(predicted_val, dtype=np.float32)
            ground_truth_np = np.asarray(ground_truth_val, dtype=np.float32)
        except Exception:
            # Error converting values to numpy arrays (e.g., non-numeric data)
            return 1.0

        if predicted_np.shape != ground_truth_np.shape:
            # Shape mismatch between predicted and ground truth arrays
            return 1.0

        violation: float
        if predicted_np.ndim == 0:  # Scalar value
            diff = np.abs(predicted_np - ground_truth_np)
            gt_mag = np.abs(ground_truth_np)
            if gt_mag < 1e-9: # Avoid division by zero for very small ground truth
                 violation = diff
            else:
                 violation = diff / gt_mag
        elif predicted_np.ndim >= 1:  # Vector or higher-dimensional tensor value
            # Flatten arrays for norm calculation to handle multi-dim quantities
            diff_norm = np.linalg.norm(predicted_np.flatten() - ground_truth_np.flatten())
            gt_norm = np.linalg.norm(ground_truth_np.flatten())

            if gt_norm < 1e-9: # If ground truth is zero or near-zero vector/tensor
                violation = diff_norm
            else:
                violation = diff_norm / gt_norm
        else:
            return 1.0

        return float(np.clip(violation, 0.0, 1.0)) # Clip violation to [0,1] as it's used in exp

    def compute_conservation_bonus(self,
                                   predicted_traj_data: Dict[str, Any], # Renamed for clarity
                                   ground_truth_traj_data: Dict[str, Any], # Renamed for clarity
                                   hypothesis_params: Dict[str, Any] # Retained but unused as per original
                                  ) -> float:
        """
        Computes a "bonus" score based on how well predicted trajectories adhere to conservation laws.

        The bonus is calculated for each specified conservation type. For each type,
        the violation between predicted and ground truth values is computed using `_calculate_violation`.
        This violation is then transformed into a bonus using an exponential decay function,
        scaled by a tolerance factor. The final bonus is the average of individual
        bonuses, multiplied by the overall `self.weight_factor`.

        Args:
            predicted_traj_data: A dictionary containing predicted conserved quantities.
                                 Expected keys are like 'conserved_energy', 'conserved_momentum'.
            ground_truth_traj_data: A dictionary containing ground truth conserved quantities.
                                  Expected keys match those in predicted_traj_data.
            hypothesis_params: Parameters of the hypothesis that generated the prediction.
                               Currently unused in this method but kept for API consistency.

        Returns:
            A float representing the total conservation bonus. Returns 0.0 if no
            conservation laws are evaluated (e.g., due to missing data for all specified types).
        """
        total_bonus_sum: float = 0.0
        num_laws_evaluated: int = 0

        for c_type in self.conservation_types:
            pred_val: Optional[Union[np.ndarray, float]] = predicted_traj_data.get(f'conserved_{c_type}')
            gt_val: Optional[Union[np.ndarray, float]] = ground_truth_traj_data.get(f'conserved_{c_type}')

            if pred_val is None or gt_val is None:
                # print(f"Warning: Missing data for conservation type '{c_type}'. Skipping.") # For debugging
                continue # Skip this conservation type if data is missing

            violation: float = self._calculate_violation(pred_val, gt_val, c_type)
            tolerance: float = self.tolerances.get(c_type, 1e-3) # Default tolerance if not specified

            # Bonus is higher for lower violation, scaled exponentially by tolerance
            bonus: float = np.exp(-violation / tolerance)
            total_bonus_sum += bonus
            num_laws_evaluated += 1

            # Store history for diagnostics (timestamp for context)
            if c_type not in self.history:
                self.history[c_type] = []
            self.history[c_type].append({'violation': violation, 'bonus': bonus, 'timestamp': time.time()})

        if num_laws_evaluated == 0:
            return 0.0 # Return 0 if no laws could be evaluated

        average_bonus: float = total_bonus_sum / num_laws_evaluated
        return self.weight_factor * average_bonus # Apply overall weight factor

    def diagnose_conservation_violations(self) -> Dict[str, Dict[str, Union[float, int]]]:
        """
        Provides diagnostic information about conservation law violations from the internal history.

        This method aggregates the history of violations and bonuses for each
        conservation type, calculating average violations, average bonuses, and
        the number of evaluations.

        Returns:
            A dictionary where keys are conservation types (e.g., 'energy').
            Each value is another dictionary containing:
                'average_violation': The mean violation recorded for that type.
                'average_bonus': The mean bonus awarded for that type.
                'num_evals': The number of times this conservation type was evaluated.
        """
        diagnostics: Dict[str, Dict[str, Union[float, int]]] = {}
        for c_type, records in self.history.items():
            if records:
                avg_violation: float = float(np.mean([r['violation'] for r in records]))
                avg_bonus: float = float(np.mean([r['bonus'] for r in records]))
                diagnostics[c_type] = {
                    'average_violation': avg_violation,
                    'average_bonus': avg_bonus,
                    'num_evals': len(records)
                }
        return diagnostics

    def get_entropy_production_penalty(self,
                                       predicted_forward_traj: Optional[Dict[str, Any]],
                                       predicted_backward_traj: Optional[Dict[str, Any]]) -> float:
        """
        Calculates a penalty based on an approximation of entropy production (irreversibility).

        This method compares the change in a conserved quantity (typically 'energy')
        over a forward trajectory with its change over a time-reversed backward trajectory.
        A larger difference (dissipation metric) implies higher entropy production,
        resulting in a more negative penalty.

        Args:
            predicted_forward_traj: Dictionary of conserved quantities from the forward pass.
                                    Expected to contain 'conserved_energy' as a list/array of values.
            predicted_backward_traj: Dictionary of conserved quantities from the backward pass
                                     (with initial and final states swapped relative to forward).
                                     Expected to contain 'conserved_energy_reversed_path'.

        Returns:
            A float representing the entropy production penalty. This value is typically
            negative or zero. Returns 0.0 if required data is missing or invalid.
        """
        if predicted_forward_traj is None or predicted_backward_traj is None:
            return 0.0

        energy_fwd: Optional[Union[List[float], np.ndarray]] = predicted_forward_traj.get('conserved_energy')
        energy_bwd: Optional[Union[List[float], np.ndarray]] = predicted_backward_traj.get('conserved_energy_reversed_path')

        if energy_fwd is not None and energy_bwd is not None:
            energy_fwd_np = np.asarray(energy_fwd, dtype=np.float32)
            energy_bwd_np = np.asarray(energy_bwd, dtype=np.float32)

            # Need at least two points to see a change in energy
            if energy_fwd_np.size < 2 or energy_bwd_np.size < 2 :
                return 0.0

            # Calculate absolute change in energy over the forward and backward paths
            fwd_change: float = float(np.abs(energy_fwd_np[-1] - energy_fwd_np[0]))
            bwd_change: float = float(np.abs(energy_bwd_np[-1] - energy_bwd_np[0]))

            # Dissipation metric: measures the asymmetry of energy change
            denominator = (fwd_change + bwd_change) / 2.0 + 1e-9 # Avoid division by zero
            dissipation_metric: float = np.abs(fwd_change - bwd_change) / denominator

            # Penalty is logarithmic with the dissipation; higher dissipation = larger negative penalty
            # Using -log(1+x) ensures penalty is <= 0.
            penalty: float = -np.log(1 + dissipation_metric)

            # The original `conservation_reward_fix.py` had an additional `* 0.1` scaling factor
            # for this penalty when returned from `get_entropy_production_penalty`.
            # We'll retain that for consistency with its original behavior.
            return penalty * 0.1 # This penalty is typically negative or zero
        return 0.0

    def check_conservation_with_evaluation(self,
                                           expr_str: str,
                                           trajectory_data: Dict[str, Any],
                                           variables: List[Variable],
                                           conservation_type: str) -> bool:
        """
        A combined check method that evaluates the expression and then calls
        the internal `_calculate_violation` to determine if it's conserved.
        This provides a single entry point for external modules (like ConservationLawReward)
        to check conservation.
        """
        if not expr_str or not trajectory_data or not variables:
            return False

        # Evaluate the expression on the trajectory data
        evaluated_quantity_values = evaluate_expression_on_data(expr_str, trajectory_data)

        if evaluated_quantity_values is None or evaluated_quantity_values.size == 0 or np.any(np.isnan(evaluated_quantity_values)):
            return False # Cannot evaluate or result is invalid

        # For simplicity, assume that `trajectory_data` contains the ground truth
        # for the 'conservation_type' directly, or that the check is against constancy.
        # Here, we check constancy against the first value of the evaluated quantity.
        ground_truth_for_constancy = np.full_like(evaluated_quantity_values, evaluated_quantity_values[0])

        violation = self._calculate_violation(evaluated_quantity_values, ground_truth_for_constancy, conservation_type)
        tolerance = self.tolerances.get(conservation_type, 1e-3)

        return violation <= tolerance


if __name__ == '__main__':
    # This __main__ block serves as a test for the ConservationBiasedReward class
    # in its role as a "law detector".

    # Mock `evaluate_expression_on_data` and `Variable` if not fully available
    try:
        from janus_ai.core.expressions.symbolic_math import evaluate_expression_on_data as real_eval_expr_on_data
    except ImportError:
        print("Using mock evaluate_expression_on_data for testing ConservationBiasedReward. Please ensure real utility exists.")
        def evaluate_expression_on_data(expr_str: str, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
            """Mocks evaluation for testing: returns constant array or simple transform."""
            if 'energy' in expr_str:
                # Simulate a nearly constant energy for "conserved" scenario
                return np.full_like(data_dict.get('x', np.array([0.0])), 10.0) + np.random.rand(data_dict.get('x', np.array([0.0])).shape[0]) * 1e-4
            elif 'momentum' in expr_str:
                return np.full_like(data_dict.get('x', np.array([0.0])), 5.0) + np.random.rand(data_dict.get('x', np.array([0.0])).shape[0]) * 1e-5
            elif 'x' in expr_str and 'x' in data_dict:
                # Simulate a varying quantity for "not conserved" scenario
                return data_dict['x'] * np.sin(data_dict.get('t', 1.0))
            return np.array([0.0]) # Default mock return

    try:
        from janus_ai.core.expressions.expression import Variable as RealVariable
    except ImportError:
        print("Using mock Variable for ConservationBiasedReward test.")
        # Minimal mock for Variable needed for testing
        @dataclass(eq=True, frozen=False)
        class RealVariable:
            name: str
            index: int
            properties: Dict[str, Any] = field(default_factory=dict)
            symbolic: Any = field(init=False)
            def __post_init__(self): self.symbolic = sp.Symbol(self.name)
            def __hash__(self): return hash((self.name, self.index))
            def __str__(self): return self.name
            @property
            def complexity(self) -> int: return 1
    Variable = RealVariable


    print("--- Testing ConservationBiasedReward as a Law Detector ---")

    reward_system_detector = ConservationBiasedReward(
        conservation_types=['energy', 'momentum'],
        weight_factor=1.0 # This weight factor is for its own bonus output, not the external reward.
    )

    # Define some dummy variables (x, v, t, m, k consistent with typical physics laws)
    var_x = Variable("x", 0)
    var_v = Variable("v", 1)
    var_t = Variable("t", 2)
    var_m = Variable("m", 3)
    var_k = Variable("k", 4)
    variables_list = [var_x, var_v, var_t, var_m, var_k]

    # Dummy trajectory data for simulating physics environments
    # Each array should have same length as `n_samples` used in evaluate_expression_on_data
    n_traj_samples = 100
    trajectory_data_mock = {
        'x': np.linspace(0, 10, n_traj_samples),
        'v': np.linspace(0, 5, n_traj_samples),
        't': np.linspace(0, 1, n_traj_samples),
        'm': np.full(n_traj_samples, 1.0), # Constant mass
        'k': np.full(n_traj_samples, 2.0)  # Constant spring constant
    }

    # Test `check_conservation_with_evaluation`
    print("\n--- Testing check_conservation_with_evaluation ---")
    energy_expr = "0.5 * m * v**2 + 0.5 * k * x**2" # Should be detected as conserved by mock
    is_energy_conserved = reward_system_detector.check_conservation_with_evaluation(
        energy_expr, trajectory_data_mock, variables_list, 'energy'
    )
    print(f"Is '{energy_expr}' conserved (via check_conservation_with_evaluation)? {is_energy_conserved}")
    assert is_energy_conserved == True, "Expected energy to be detected as conserved."

    momentum_expr = "m * v" # Should be detected as conserved by mock
    is_momentum_conserved = reward_system_detector.check_conservation_with_evaluation(
        momentum_expr, trajectory_data_mock, variables_list, 'momentum'
    )
    print(f"Is '{momentum_expr}' conserved (via check_conservation_with_evaluation)? {is_momentum_conserved}")
    assert is_momentum_conserved == True, "Expected momentum to be detected as conserved."

    varying_expr = "x * v" # Should *not* be detected as conserved
    is_varying_conserved = reward_system_detector.check_conservation_with_evaluation(
        varying_expr, trajectory_data_mock, variables_list, 'energy' # Type doesn't matter much for non-conserved
    )
    print(f"Is '{varying_expr}' conserved? {is_varying_conserved}")
    assert is_varying_conserved == False, "Expected varying expression to not be detected as conserved."


    # Test `compute_conservation_bonus`
    print("\n--- Testing compute_conservation_bonus ---")
    # For this, we need *predicted* and *ground_truth* conserved quantities.
    # The evaluation above gives us the predicted values from the expression.
    # For ground truth, we'll use a perfectly constant array based on the first predicted value.

    # Evaluate energy and momentum expressions to get "predicted" conserved quantities
    predicted_energy_values = evaluate_expression_on_data(energy_expr, trajectory_data_mock)
    predicted_momentum_values = evaluate_expression_on_data(momentum_expr, trajectory_data_mock)

    mock_predicted_traj = {
        'conserved_energy': predicted_energy_values,
        'conserved_momentum': predicted_momentum_values,
    }
    mock_ground_truth_traj = {
        'conserved_energy': np.full_like(predicted_energy_values, predicted_energy_values[0]),
        'conserved_momentum': np.full_like(predicted_momentum_values, predicted_momentum_values[0]),
    }

    conservation_bonus = reward_system_detector.compute_conservation_bonus(
        predicted_traj_data=mock_predicted_traj,
        ground_truth_traj_data=mock_ground_truth_traj,
        hypothesis_params={}
    )
    print(f"Computed Conservation Bonus: {conservation_bonus:.4f}")
    assert conservation_bonus > 0.9 * reward_system_detector.weight_factor, "Expected high bonus for conserved quantities."


    # Test `get_entropy_production_penalty`
    print("\n--- Testing Entropy Production Penalty ---")
    # Simulate energy values for forward and backward trajectories
    fwd_traj_energy = {'conserved_energy': np.array([10.0, 9.8, 9.6, 9.4])} # Energy decreases by 0.6
    bwd_traj_ideal = {'conserved_energy_reversed_path': np.array([9.4, 9.6, 9.8, 10.0])} # Perfect reversal, change of 0.6
    bwd_traj_dissipated = {'conserved_energy_reversed_path': np.array([9.4, 9.5, 9.6, 9.7])} # Dissipated reversal, change of 0.3

    penalty_ideal = reward_system_detector.get_entropy_production_penalty(fwd_traj_energy, bwd_traj_ideal)
    penalty_dissipated = reward_system_detector.get_entropy_production_penalty(fwd_traj_energy, bwd_traj_dissipated)

    print(f"Penalty (ideal reversal): {penalty_ideal:.4f} (should be near 0)")
    print(f"Penalty (dissipated reversal): {penalty_dissipated:.4f} (should be negative)")
    assert penalty_ideal >= -1e-6 # Should be effectively zero or very close
    assert penalty_dissipated < -0.01 # Should be a noticeable negative penalty


    # Test diagnostics
    print("\n--- Testing Diagnostics ---")
    diagnostics = reward_system_detector.diagnose_conservation_violations()
    print(diagnostics)
    assert 'energy' in diagnostics
    assert 'momentum' in diagnostics
    assert diagnostics['energy']['num_evals'] >= 1
    assert diagnostics['momentum']['num_evals'] >= 1


    print("\nAll ConservationBiasedReward (as Law Detector) tests completed.")
