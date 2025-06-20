"""
CurriculumManager
=================

Manages the curriculum learning process, progressively increasing the difficulty
of tasks presented to the agent.
"""

from typing import Dict, Any, Optional
import time
import numpy as np

# Import the task distribution from its new location
from janus.physics.data.generators import PhysicsTaskDistribution


class CurriculumManager:
    """
    Manages the curriculum learning process by controlling the difficulty
    of tasks sampled from a PhysicsTaskDistribution.

    This manager orchestrates the progression of learning by
    adjusting internal parameters that influence how tasks are sampled,
    moving from simpler to more complex tasks over time.
    """

    def __init__(self, 
                 task_distribution: PhysicsTaskDistribution,
                 initial_difficulty: float = 0.0,
                 max_difficulty: float = 1.0,
                 difficulty_increment: float = 0.05,
                 steps_per_increment: int = 100,
                 adaptive: bool = False, # Flag for adaptive curriculum (future extension)
                 patience: int = 5, # For adaptive curriculum: how many steps without improvement before adjusting
                 min_performance_threshold: float = 0.7 # For adaptive: min performance to advance
                ):
        """
        Initializes the CurriculumManager.

        Args:
            task_distribution: The PhysicsTaskDistribution instance responsible for sampling tasks.
                               The manager will update its internal difficulty state.
            initial_difficulty: The starting difficulty level (0.0 to 1.0).
            max_difficulty: The maximum difficulty level to reach.
            difficulty_increment: How much to increase difficulty per step.
            steps_per_increment: Number of curriculum steps before difficulty is increased.
            adaptive: If True, the curriculum advances based on performance (future work).
            patience: How many steps to wait for improvement in adaptive mode.
            min_performance_threshold: Performance threshold to advance curriculum in adaptive mode.
        """
        self.task_distribution = task_distribution
        self.current_difficulty = initial_difficulty
        self.max_difficulty = max_difficulty
        self.difficulty_increment = difficulty_increment
        self.steps_per_increment = steps_per_increment
        self.adaptive = adaptive

        self.current_step = 0
        self.epochs_since_last_increment = 0

        # For adaptive curriculum (future extension)
        self.patience = patience
        self.min_performance_threshold = min_performance_threshold
        self.best_performance = -float('inf')
        self.steps_without_improvement = 0

        # Set initial difficulty on the task distribution
        self._update_task_distribution_difficulty()
        print(f"CurriculumManager initialized. Starting difficulty: {self.current_difficulty:.2f}")

    def _update_task_distribution_difficulty(self):
        """
        Updates the difficulty parameter within the linked PhysicsTaskDistribution.
        This method assumes that PhysicsTaskDistribution has a settable `current_difficulty`
        or a similar mechanism. If not, this method would need to call a specific
        method on `task_distribution` to communicate the desired difficulty.
        """
        # This is a conceptual link. The PhysicsTaskDistribution needs to expose a way
        # to set its internal "current_difficulty_level" that `sample_task_batch` then uses.
        if hasattr(self.task_distribution, 'set_difficulty_level'):
            self.task_distribution.set_difficulty_level(self.current_difficulty)
        else:
            # Fallback or warning: if task_distribution doesn't support setting difficulty,
            # its `sample_task_batch(curriculum=True)` implicitly manages it.
            # In this case, the CurriculumManager primarily serves as a tracker.
            pass # The PhysicsTaskDistribution.sample_task_batch(curriculum=True) handles it.

    def step(self, performance_metrics: Optional[Dict[str, float]] = None):
        """
        Advances the curriculum.

        In non-adaptive mode, difficulty increases every `steps_per_increment` steps.
        In adaptive mode (future work), difficulty increases based on performance.

        Args:
            performance_metrics: Dictionary of performance metrics from recent training,
                                 used in adaptive curriculum. E.g., {'mean_reward': X}.
        """
        self.current_step += 1
        self.epochs_since_last_increment += 1

        if self.adaptive:
            self._adaptive_step(performance_metrics)
        else:
            self._fixed_step()

    def _fixed_step(self):
        """Advances the curriculum based on fixed intervals."""
        if self.epochs_since_last_increment >= self.steps_per_increment:
            if self.current_difficulty < self.max_difficulty:
                self.current_difficulty = min(self.max_difficulty, self.current_difficulty + self.difficulty_increment)
                self.epochs_since_last_increment = 0 # Reset counter
                self._update_task_distribution_difficulty()
                print(f"Curriculum advanced to difficulty: {self.current_difficulty:.2f} at step {self.current_step}")

    def _adaptive_step(self, performance_metrics: Optional[Dict[str, float]]):
        """
        Advances the curriculum adaptively based on performance (future extension).
        This method is a placeholder and would require more sophisticated logic.
        """
        if performance_metrics and 'mean_reward' in performance_metrics:
            current_performance = performance_metrics['mean_reward']
            
            if current_performance > self.best_performance:
                self.best_performance = current_performance
                self.steps_without_improvement = 0
            else:
                self.steps_without_improvement += 1

            # Advance if performance is good enough and we've waited long enough
            if current_performance >= self.min_performance_threshold and \
               self.epochs_since_last_increment >= self.steps_per_increment:
                
                if self.current_difficulty < self.max_difficulty:
                    self.current_difficulty = min(self.max_difficulty, self.current_difficulty + self.difficulty_increment)
                    self.epochs_since_last_increment = 0
                    self.steps_without_improvement = 0
                    self._update_task_distribution_difficulty()
                    print(f"Adaptive curriculum advanced to difficulty: {self.current_difficulty:.2f} at step {self.current_step} due to performance.")
            
            elif self.steps_without_improvement >= self.patience:
                print(f"Adaptive curriculum stuck: No improvement for {self.patience} steps. Consider adjusting parameters or curriculum strategy.")
                # Could implement strategies like:
                # - Decreasing difficulty slightly (curriculum retreat)
                # - Increasing exploration
                # - Logging a warning and continuing without advancing
        else:
            # If no performance metrics are provided in adaptive mode, warn or raise error
            print("Warning: Adaptive curriculum requires 'performance_metrics' but none were provided.")
            # For now, just continue without advancing.

    def get_current_difficulty(self) -> float:
        """Returns the current curriculum difficulty level."""
        return self.current_difficulty

    def reset(self):
        """Resets the curriculum to its initial state."""
        self.current_difficulty = 0.0
        self.current_step = 0
        self.epochs_since_last_increment = 0
        self.best_performance = -float('inf')
        self.steps_without_improvement = 0
        self._update_task_distribution_difficulty()
        print("CurriculumManager reset to initial difficulty: 0.0")

    def is_finished(self) -> bool:
        """Checks if the curriculum has reached its maximum difficulty."""
        return self.current_difficulty >= self.max_difficulty

    def get_state(self) -> Dict[str, Any]:
        """Returns the current state of the curriculum manager for checkpointing."""
        return {
            'current_difficulty': self.current_difficulty,
            'current_step': self.current_step,
            'epochs_since_last_increment': self.epochs_since_last_increment,
            'best_performance': self.best_performance,
            'steps_without_improvement': self.steps_without_improvement
        }

    def set_state(self, state: Dict[str, Any]):
        """Restores the state of the curriculum manager from a saved state."""
        self.current_difficulty = state.get('current_difficulty', self.current_difficulty)
        self.current_step = state.get('current_step', self.current_step)
        self.epochs_since_last_increment = state.get('epochs_since_last_increment', self.epochs_since_last_increment)
        self.best_performance = state.get('best_performance', self.best_performance)
        self.steps_without_improvement = state.get('steps_without_improvement', self.steps_without_improvement)
        self._update_task_distribution_difficulty()
        print(f"CurriculumManager state restored. Current difficulty: {self.current_difficulty:.2f}")


if __name__ == "__main__":
    # This __main__ block demonstrates how to use the CurriculumManager.

    # Mock PhysicsTaskDistribution for testing purposes
    class MockPhysicsTaskDistribution:
        def __init__(self, include_noise: bool = True):
            self.include_noise = include_noise
            self._difficulty_level = 0.0 # Internal state for difficulty
            self.tasks = {
                0.0: ["simple_pendulum", "free_fall"],
                0.2: ["harmonic_oscillator"],
                0.5: ["double_pendulum", "projectile_motion_with_drag"],
                0.8: ["n_body_problem"],
                1.0: ["quantum_harmonic_oscillator"]
            }
            # Simulate a method that real PhysicsTaskDistribution might have
            self.set_difficulty_level(self._difficulty_level)

        def set_difficulty_level(self, level: float):
            """Sets the internal difficulty level."""
            self._difficulty_level = level
            print(f"MockPhysicsTaskDistribution internal difficulty set to: {self._difficulty_level:.2f}")

        def sample_task_batch(self, batch_size: int, curriculum: bool = True) -> List[Any]:
            """
            Samples tasks based on the current difficulty level.
            In a real scenario, this would dynamically generate or select tasks.
            """
            sampled_tasks = []
            available_tasks = []
            for diff_level, task_list in self.tasks.items():
                if diff_level <= self._difficulty_level:
                    available_tasks.extend(task_list)
            
            if not available_tasks: # If no tasks available yet, return empty list or default
                print("No tasks available at current difficulty level. Returning empty batch.")
                return []

            # Simple sampling: pick randomly from available tasks
            for _ in range(batch_size):
                sampled_tasks.append(np.random.choice(available_tasks))
            
            return sampled_tasks

        def describe_task_distribution(self):
            return "Mock Physics Task Distribution for testing curriculum."

    print("--- CurriculumManager Demonstration (Fixed Schedule) ---")
    mock_task_dist = MockPhysicsTaskDistribution()
    
    # Initialize CurriculumManager for fixed schedule
    curriculum_fixed = CurriculumManager(
        task_distribution=mock_task_dist,
        initial_difficulty=0.0,
        max_difficulty=1.0,
        difficulty_increment=0.2,
        steps_per_increment=3, # Increment every 3 steps for demo
        adaptive=False
    )

    # Simulate training steps
    for i in range(15):
        print(f"\n--- Training Step {i+1} ---")
        
        # In a real scenario, you would call `trainer.train_step()` or `trainer.collect_rollouts()` here.
        # Then you'd get performance metrics from the trainer.
        
        # Advance the curriculum manager
        curriculum_fixed.step()
        
        # Sample tasks for this step (the manager implicitly updates the distribution)
        current_tasks = mock_task_dist.sample_task_batch(batch_size=2, curriculum=True)
        print(f"Current curriculum difficulty: {curriculum_fixed.get_current_difficulty():.2f}")
        print(f"Sampled tasks: {current_tasks}")
        time.sleep(0.1) # Simulate some work

    print("\nCurriculum finished (fixed schedule):", curriculum_fixed.is_finished())
    

    print("\n--- CurriculumManager Demonstration (Adaptive Schedule - Placeholder) ---")
    mock_task_dist_adaptive = MockPhysicsTaskDistribution()
    curriculum_adaptive = CurriculumManager(
        task_distribution=mock_task_dist_adaptive,
        initial_difficulty=0.0,
        max_difficulty=1.0,
        difficulty_increment=0.2,
        steps_per_increment=2, # Check performance every 2 steps
        adaptive=True,
        patience=3, # If performance doesn't improve for 3 checks, print warning
        min_performance_threshold=0.8 # Need 80% mean_reward to advance
    )

    performance_history = [0.7, 0.75, 0.82, 0.85, 0.78, 0.90, 0.91, 0.88, 0.95, 0.96]

    for i, perf in enumerate(performance_history):
        print(f"\n--- Training Step {i+1} (Adaptive) ---")
        metrics = {'mean_reward': perf}
        curriculum_adaptive.step(metrics)
        print(f"Current curriculum difficulty: {curriculum_adaptive.get_current_difficulty():.2f}")
        print(f"Simulated performance: {perf:.2f}")
        time.sleep(0.1)

    print("\nCurriculum finished (adaptive schedule):", curriculum_adaptive.is_finished())

    print("\n--- Testing Checkpointing ---")
    curriculum_checkpoint_test = CurriculumManager(mock_task_dist, initial_difficulty=0.1)
    curriculum_checkpoint_test.step()
    curriculum_checkpoint_test.step()
    print(f"Curriculum state before saving: {curriculum_checkpoint_test.get_state()}")
    saved_state = curriculum_checkpoint_test.get_state()

    print("\nCreating new manager and loading state...")
    new_curriculum_manager = CurriculumManager(mock_task_dist, initial_difficulty=0.0) # Start from scratch
    new_curriculum_manager.set_state(saved_state)
    print(f"Curriculum state after loading: {new_curriculum_manager.get_state()}")
    print("Checkpointing test complete.")
