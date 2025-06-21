# JanusAI/integration/__init__.py
"""
JanusAI Multi-Agent Integration Package
=======================================

This package integrates the multi-agent system, tiered communication architecture,
and advanced training pipelines into the core JanusAI framework.

It provides the necessary components to run collaborative scientific discovery
experiments using a society of specialized AI agents.

Primary Entry Points:
 - `run_training_pipeline`: Main function to start a training run.
 - `AdvancedJanusTrainer`: The core orchestrator for multi-agent training.
 - `DiscoveryAgent`: The base class for creating new discovery agents.

Core Schemas:
 - `Discovery`: Dataclass representing a validated scientific finding.
 - `Message`: Dataclass for inter-agent communication packets.
 - `AgentRole`, `MessageType`: Enums for defining agent and message types.

Example Usage:
    from janus_ai.integration import run_training_pipeline, AdvancedJanusTrainer
    from janus_ai.config import JanusConfig

    # Configure and run a multi-agent experiment
    config = JanusConfig(training_mode="multi_agent", ...)
    trainer = run_training_pipeline(config)

Version: 1.0.0
Author: JanusAI Team
License: MIT
"""

import logging

__version__ = "1.0.0"
__author__ = "JanusAI Team"

# --- Public API for the Integration Package ---

# Core data structures for type hinting and message creation
from janus_ai.integration.schemas import (
    AgentRole,
    MessageType,
    Discovery,
    Message
)

# Core infrastructure for advanced use cases
from janus_ai.integration.knowledge import (
    MessageBus,
    SharedKnowledgeBase
)

# Core agent and communication modules
from janus_ai.integration.agent import (
    DiscoveryAgent,
    CommunicationEncoder,
    CommunicationAggregator
)

# Primary pipeline and trainer entry points
from janus_ai.integration.pipeline import (
    JanusTrainer,
    AdvancedJanusTrainer,
    create_trainer,
    run_training_pipeline,
    run_multi_agent_discovery
)

# Expose sub-modules for advanced, direct import
from janus_ai.integration import distributed
from janus_ai.integration import meta_learning


# Define what is exposed when a user does 'from janus_ai.integration import *'
__all__ = [
    # High-level pipeline functions
    "run_training_pipeline",
    "run_multi_agent_discovery",
    "create_trainer",

    # Core Trainer and Agent classes
    "JanusTrainer",
    "AdvancedJanusTrainer",
    "DiscoveryAgent",

    # Core Schemas
    "AgentRole",
    "MessageType",
    "Discovery",
    "Message",

    # Infrastructure components (for advanced customization)
    "MessageBus",
    "SharedKnowledgeBase",

    # Neural communication modules
    "CommunicationEncoder",
    "CommunicationAggregator",
    
    # Sub-modules for advanced usage
    "distributed",
    "meta_learning",

    # Package metadata
    "__version__",
    "__author__",
]

# Set up a null handler to avoid 'No handler found' warnings.
# This allows the end-user application to configure logging.
logging.getLogger(__name__).addHandler(logging.NullHandler())
