"""
Tests for experiments/configs/validation_suites.py
"""
import pytest
from experiments.configs.validation_suites import ValidationSuite, ValidationSuiteLibrary

# --- Tests for ValidationSuite ---
class TestValidationSuite:
    def test_default_initialization(self):
        suite = ValidationSuite(name="Test Suite", description="A test description.")
        assert suite.name == "Test Suite"
        assert suite.description == "A test description."
        assert suite.evaluation_tasks == []
        assert suite.metrics_to_report == ["average_fidelity", "correct_discovery_rate"]
        assert suite.success_criteria == {}
        assert suite.eval_config_overrides == {}

    def test_custom_initialization(self):
        tasks = ["task1", "task2"]
        metrics = ["metric_a", "metric_b"]
        criteria = {"metric_a": 0.9}
        overrides = {"param": "value"}
        suite = ValidationSuite(
            name="Custom Suite",
            description="Custom desc.",
            evaluation_tasks=tasks,
            metrics_to_report=metrics,
            success_criteria=criteria,
            eval_config_overrides=overrides
        )
        assert suite.name == "Custom Suite"
        assert suite.description == "Custom desc."
        assert suite.evaluation_tasks == tasks
        assert suite.metrics_to_report == metrics
        assert suite.success_criteria == criteria
        assert suite.eval_config_overrides == overrides

# --- Tests for ValidationSuiteLibrary ---
class TestValidationSuiteLibrary:
    @pytest.fixture
    def library(self):
        return ValidationSuiteLibrary()

    def test_init(self, library):
        assert len(library._suites) > 0  # Should have default suites
        assert len(library._suites_by_name) == len(library._suites)
        for suite in library._suites:
            assert library._suites_by_name[suite.name] == suite

    def test_create_default_suites(self, library):
        # This method is called by __init__, so we primarily check its output via the library instance
        default_suites = library._create_default_suites() # Can also call directly for isolation if needed
        assert isinstance(default_suites, list)
        assert len(default_suites) >= 3 # Based on current implementation (Basic, Advanced, AI Fidelity)

        for suite in default_suites:
            assert isinstance(suite, ValidationSuite)
            assert suite.name is not None
            assert suite.description is not None

        # Check properties of a known default suite
        basic_physics_suite = next((s for s in default_suites if s.name == "Basic Physics Discovery"), None)
        assert basic_physics_suite is not None
        assert "harmonic_oscillator_energy" in basic_physics_suite.evaluation_tasks
        assert basic_physics_suite.success_criteria == {"correct_discovery_rate": 0.7, "average_fidelity": 0.85}

    def test_get_suite_by_name(self, library):
        # Test existing suite
        basic_suite = library.get_suite_by_name("Basic Physics Discovery")
        assert basic_suite is not None
        assert basic_suite.name == "Basic Physics Discovery"

        # Test non-existent suite
        non_existent_suite = library.get_suite_by_name("Non Existent Suite")
        assert non_existent_suite is None

    def test_get_all_suites(self, library):
        all_suites = library.get_all_suites()
        assert isinstance(all_suites, list)
        assert len(all_suites) == len(library._suites) # Should match internal list
        for suite in all_suites:
            assert isinstance(suite, ValidationSuite)

    def test_describe_suites(self, library):
        description_string = library.describe_suites()
        assert isinstance(description_string, str)
        assert f"Validation Suite Library ({len(library._suites)} suites):" in description_string

        # Check if names and descriptions of default suites are present
        for suite in library._suites:
            assert suite.name in description_string
            assert suite.description in description_string
            if suite.evaluation_tasks:
                assert "Tasks:" in description_string
                assert suite.evaluation_tasks[0] in description_string # Check at least one task
            assert "Metrics:" in description_string
            assert suite.metrics_to_report[0] in description_string # Check at least one metric
            assert "Criteria:" in description_string
            if suite.success_criteria:
                 # Check one key-value pair from success_criteria
                first_crit_key = list(suite.success_criteria.keys())[0]
                assert f"'{first_crit_key}': {suite.success_criteria[first_crit_key]}" in description_string
