from typing import Any, Dict
import numpy as np
import torch

from JanusAI.ai_interpretability.evaluation.fidelity import ModelFidelityEvaluator

class InterpretabilityReward:
    # existing class content...

    def _calculate_fidelity(self,
                            expression: Any,
                            ai_model: Any,
                            test_data: Any) -> float:
        """
        Calculate how well the symbolic expression reproduces the AI model's attention behavior.
        Delegates to ModelFidelityEvaluator for robust, normalized fidelity scoring.
        """
        try:
            # Initialize calculator once
            # TODO: ModelFidelityEvaluator requires ai_model, data_samples, variables at init.
            # This initialization needs to be refactored.
            if not hasattr(self, '_fidelity_calculator') or self._fidelity_calculator is None:
                # Placeholder initialization, will likely need adjustment by user/further refactoring
                # as ModelFidelityEvaluator has a different constructor.
                self._fidelity_calculator = ModelFidelityEvaluator(ai_model=ai_model, data_samples=test_data, variables=getattr(self, 'variables', []))


            # Normalize test_data into dict format expected by ModelFidelityEvaluator
            if hasattr(test_data, 'inputs') and hasattr(test_data, 'attention_weights'):
                data_dict = {
                    'input_ids': np.array(test_data.inputs),
                    'attention_mask': np.array(getattr(test_data, 'attention_mask', np.ones_like(test_data.inputs))),
                    'sequence_length': test_data.inputs.shape[-1],
                    'target_layer': getattr(test_data, 'target_layer', 0),
                    'target_head': getattr(test_data, 'target_head', None)
                }
            elif isinstance(test_data, dict):
                data_dict = test_data
            else:
                raise ValueError(f"Unrecognized test_data format: {type(test_data)}")

            # Use variables attribute if present
            variables = getattr(self, 'variables', [])

            return self._fidelity_calculator.calculate_fidelity(
                expression=expression,
                ai_model=ai_model,
                test_data=data_dict,
                variables=variables,
                target_behavior='attention'
            )
        except Exception as e:
            # Log error if logger available, else print
            try:
                self.logger.error(f"Fidelity calculation error: {e}", exc_info=True)
            except Exception:
                print(f"Error in _calculate_fidelity: {e}")
            # Return worst-case fidelity
            return 0.0
