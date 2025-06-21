from typing import Any, Dict
import numpy as np
import torch

from JanusAI.ai_interpretability.evaluation.fidelity import ModelFidelityEvaluator

class InterpretabilityReward:
    # existing class content...

    def _calculate_fidelity(self, predicted: np.ndarray, target: np.ndarray) -> float:
        """MSE in attention probability space"""
        # Ensure inputs are numpy arrays, if not already.
        # This might be redundant if type hints are strictly followed by callers,
        # but good for robustness.
        predicted = np.asarray(predicted)
        target = np.asarray(target)

        if predicted.shape != target.shape:
            # Handle or log shape mismatch if necessary
            # For now, let np.mean raise an error or broadcast if applicable,
            # though for MSE, shapes should ideally match.
            # Consider raising a ValueError here if strict shape matching is required.
            print(f"Warning: Shape mismatch in _calculate_fidelity. Predicted: {predicted.shape}, Target: {target.shape}")

        mse = np.mean((predicted - target) ** 2)
        return 1.0 / (1.0 + mse)

    # Any other methods of the class would follow here...
