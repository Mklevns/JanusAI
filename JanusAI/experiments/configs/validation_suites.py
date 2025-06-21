"""
Experiment Validation Suites
============================

Defines predefined validation suites and configurations for experiments.
These suites can be used to set up standard evaluation protocols or
to validate specific aspects of discovered laws or model behavior.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

# Import ExperimentConfig for type hinting
from janus_ai.experiments.configs.experiment_config import ExperimentConfig


@dataclass
class ValidationSuite:
    """
    A data class representing a predefined validation suite.
    A validation suite specifies a set of evaluation tasks, metrics,
    and criteria for assessing experiment outcomes.
    """
    name: str
    description: str
    evaluation_tasks: List[str] = field(default_factory=list) # List of task names from PhysicsTaskDistribution
    metrics_to_report: List[str] = field(default_factory=lambda: ["average_fidelity", "correct_discovery_rate"])
    success_criteria: Dict[str, float] = field(default_factory=dict) # e.g., {"correct_discovery_rate": 0.8}
    eval_config_overrides: Dict[str, Any] = field(default_factory=dict) # Overrides for default evaluation settings


class ValidationSuiteLibrary:
    """
    A collection of predefined validation suites.
    """
    def __init__(self):
        self._suites: List[ValidationSuite] = self._create_default_suites()
        self._suites_by_name: Dict[str, ValidationSuite] = {suite.name: suite for suite in self._suites}

    def _create_default_suites(self) -> List[ValidationSuite]:
        """
        Populates the library with common validation suites.
        """
        suites = [
            ValidationSuite(
                name="Basic Physics Discovery",
                description="Evaluates discovery of fundamental mechanics laws.",
                evaluation_tasks=[
                    "harmonic_oscillator_energy",
                    "Newton's Second Law",  # Assuming this task exists in PhysicsTaskDistribution
                    "pendulum_small_angle",
                    "ideal_gas_law",
                ],
                success_criteria={
                    "correct_discovery_rate": 0.7,
                    "average_fidelity": 0.85,
                },
                eval_config_overrides={
                    "n_eval_samples_per_task": 500,
                    "n_task_instances_per_benchmark": 3,
                },
            )
        ]

        # Advanced Physics Discovery Suite
        suites.append(ValidationSuite(
            name="Advanced Physics Discovery",
            description="Evaluates discovery of more complex or composite physical laws.",
            evaluation_tasks=[
                "kepler_orbit",
                "elastic_collision",
                "pendulum_nonlinear",
                "double_pendulum_energy"
            ],
            success_criteria={"correct_discovery_rate": 0.5, "average_fidelity": 0.75},
            eval_config_overrides={"n_eval_samples_per_task": 1000, "n_task_instances_per_benchmark": 5}
        ))

        # AI Interpretability Fidelity Suite
        suites.append(ValidationSuite(
            name="AI Fidelity Evaluation",
            description="Evaluates how well symbolic explanations mimic AI model behavior.",
            evaluation_tasks=[], # Tasks here would typically specify AI models/datasets
            metrics_to_report=["average_fidelity", "average_simplicity", "average_consistency"],
            success_criteria={"average_fidelity": 0.9, "average_simplicity": 0.7},
            eval_config_overrides={"interpretability_metric": "r_squared"}
        ))

        return suites

    def get_suite_by_name(self, name: str) -> Optional[ValidationSuite]:
        """Retrieves a validation suite by its name."""
        return self._suites_by_name.get(name)

    def get_all_suites(self) -> List[ValidationSuite]:
        """Retrieves all predefined validation suites."""
        return list(self._suites)

    def describe_suites(self) -> str:
        """Returns a string summary of the validation suites in the library."""
        summary = [f"Validation Suite Library ({len(self._suites)} suites):"]
        for suite in self._suites:
            summary.append(f"- {suite.name}: {suite.description}")
            summary.append(f"  Tasks: {', '.join(suite.evaluation_tasks) if suite.evaluation_tasks else 'N/A'}")
            summary.append(f"  Metrics: {', '.join(suite.metrics_to_report)}")
            summary.append(f"  Criteria: {suite.success_criteria}")
        return "\n".join(summary)


if __name__ == "__main__":
    print("--- Testing ValidationSuite Library ---")

    suite_library = ValidationSuiteLibrary()

    print("\n--- All Validation Suites ---")
    print(suite_library.describe_suites())

    # Test retrieving a specific suite
    basic_suite = suite_library.get_suite_by_name("Basic Physics Discovery")
    print(f"\nRetrieved suite 'Basic Physics Discovery': {basic_suite.name}")
    assert basic_suite is not None
    assert "harmonic_oscillator_energy" in basic_suite.evaluation_tasks

    ai_fidelity_suite = suite_library.get_suite_by_name("AI Fidelity Evaluation")
    print(f"\nRetrieved suite 'AI Fidelity Evaluation': {ai_fidelity_suite.name}")
    assert ai_fidelity_suite is not None
    assert "average_simplicity" in ai_fidelity_suite.metrics_to_report

    # Test creating a custom suite (not stored in library, but demonstrates structure)
    custom_suite = ValidationSuite(
        name="My Custom Suite",
        description="A personalized validation test.",
        evaluation_tasks=["some_custom_task"],
        metrics_to_report=["custom_metric"],
        success_criteria={"custom_metric": 0.99}
    )
    print(f"\nCustom Suite created: {custom_suite.name}")
    assert custom_suite.metrics_to_report == ["custom_metric"]

    print("\nAll ValidationSuiteLibrary tests completed.")
