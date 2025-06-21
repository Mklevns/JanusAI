from .hypothesis_generator import HypothesisGenerator
from .integration import JanusMultiAgentFramework, MultiAgentEnvironmentWrapper
from .iterative_refinement import IterativeRefinementLoop, JudgeAgent as RefinementJudgeAgent, IterationMetrics
from .multi_agent_system import (
    BaseScientificAgent,
    DynamicAgentPool,
    # PlannerAgent, # Not found in the provided snippet of multi_agent_system.py
    AgentRole,
    AgentConfig,
    HypothesisGeneratorAgent as MASystemHypothesisGeneratorAgent,
    ExperimenterAgent as MASystemExperimenterAgent,
    TheoristAgent as MASystemTheoristAgent,
    ValidatorAgent as MASystemValidatorAgent,
    CriticAgent as MASystemCriticAgent
)
from .task_setter import TaskSetterAgent, TaskSetterConfig, TaskSetterEnv
from .xolver_scientific_agents import (
    ScientificAgent as XolverScientificAgent,
    HypothesisGeneratorAgent as XolverHypothesisGeneratorAgent,
    ExperimentDesignerAgent as XolverExperimentDesignerAgent,
    SymbolicReasonerAgent as XolverSymbolicReasonerAgent,
    ValidationAgent as XolverValidationAgent,
    JudgeAgent as XolverJudgeAgent,
    XolverScientificDiscoverySystem,
    ScientificDiscovery
)

__all__ = [
    "HypothesisGenerator",
    "JanusMultiAgentFramework",
    "MultiAgentEnvironmentWrapper",
    "IterativeRefinementLoop",
    "RefinementJudgeAgent",
    "IterationMetrics",
    "BaseScientificAgent",
    "DynamicAgentPool",
    # "PlannerAgent",
    "AgentRole",
    "AgentConfig",
    "MASystemHypothesisGeneratorAgent",
    "MASystemExperimenterAgent",
    "MASystemTheoristAgent",
    "MASystemValidatorAgent",
    "MASystemCriticAgent",
    "TaskSetterAgent",
    "TaskSetterConfig",
    "TaskSetterEnv",
    "XolverScientificAgent",
    "XolverHypothesisGeneratorAgent",
    "XolverExperimentDesignerAgent",
    "XolverSymbolicReasonerAgent",
    "XolverValidationAgent",
    "XolverJudgeAgent",
    "XolverScientificDiscoverySystem",
    "ScientificDiscovery",
]
