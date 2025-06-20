"""
BaseReward
==========

Defines the abstract base class for all reward components in the Janus framework.
All specific reward implementations (e.g., intrinsic, interpretability) should
inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseReward(ABC):
    """
    Abstract base class for reward functions.

    Each concrete reward class must implement the `calculate_reward` method.
    It can also have a `weight` attribute to scale its contribution when
    multiple rewards are combined.
    """

    def __init__(self, weight: float = 1.0):
        """
        Initializes the base reward with an optional weight.

        Args:
            weight: The scalar multiplier for this reward component's value.
        """
        if not isinstance(weight, (int, float)) or weight < 0:
            raise ValueError("Reward weight must be a non-negative number.")
        self.weight = weight

    @abstractmethod
    def calculate_reward(self,
                         current_observation: Any,
                         action: Any,
                         next_observation: Any,
                         reward_from_env: float,
                         done: bool,
                         info: Dict[str, Any]) -> float:
        """
        Calculates a reward value based on the given transition and environment info.

        This method must be implemented by all concrete reward classes.
        It should return the calculated reward, which will then be scaled
        by `self.weight` when aggregated by a higher-level reward combiner.

        Args:
            current_observation: The observation before the action was taken.
            action: The action taken by the agent.
            next_observation: The observation after the action was taken.
            reward_from_env: The scalar reward returned directly by the environment (extrinsic).
                             This allows intrinsic rewards to modify or augment it.
            done: A boolean indicating if the episode terminated.
            info: A dictionary containing additional information from the environment,
                  often including the discovered expression, trajectory data, etc.

        Returns:
            The calculated reward value (unweighted by this class's `self.weight`).
        """
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> float:
        """
        Makes the reward callable, automatically applying its configured weight
        to the result of `calculate_reward`.
        This provides a consistent interface for combining rewards.
        """
        return self.calculate_reward(*args, **kwargs) * self.weight

    def describe(self) -> str:
        """Returns a brief description of the reward component."""
        return f"{self.__class__.__name__}(weight={self.weight})"

