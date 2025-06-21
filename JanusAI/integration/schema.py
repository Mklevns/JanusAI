# JanusAI/integration/schema.py
# JanusAI multi-agent system data structures
"""
Improved Multi-Agent System Data Structures for JanusAI
=======================================================

This module defines the foundational data structures for the JanusAI multi-agent
discovery system with improved type safety.

Improvements:
1. Added Union type for expression field to improve type safety
2. Added type alias for Expression type
3. Enhanced documentation

Classes:
    AgentRole: Enumeration of specialized agent roles
    MessageType: Types of messages in the communication protocol
    Discovery: Represents a validated scientific discovery
    Message: Communication packet between agents

Author: JanusAI Team
Date: 2024
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

# Type alias for expressions - can be extended based on your needs
# For example, if using SymPy: Union[str, sympy.Expr, sympy.Symbol]
Expression = Union[str, Dict[str, Any]]  # String or structured representation


class AgentRole(Enum):
    """
    Specialized roles for discovery agents in the multi-agent system.
    
    Each role represents a different exploration strategy and behavior pattern:
    - EXPLORER: Broad search with high exploration rate
    - REFINER: Focuses on improving existing discoveries
    - VALIDATOR: Verifies and validates proposed discoveries
    - SPECIALIST: Domain-specific or targeted search strategies
    """
    EXPLORER = "explorer"
    REFINER = "refiner"
    VALIDATOR = "validator"
    SPECIALIST = "specialist"


class MessageType(Enum):
    """
    Types of messages in the tiered communication protocol.
    
    The communication system uses two tiers:
    Tier 1 (Strategic):
        - DISCOVERY_POINTER: Reference to major validated discovery
        - VALIDATION_REQUEST: Request for discovery validation
        - VALIDATION_VOTE: Vote on a pending discovery
    
    Tier 2 (Tactical):
        - TACTICAL_VECTOR: Real-time coordination via latent vectors
    """
    DISCOVERY_POINTER = "discovery_pointer"    # Tier 1: Major discovery reference
    TACTICAL_VECTOR = "tactical_vector"        # Tier 2: Real-time coordination
    VALIDATION_REQUEST = "validation_request"  # Tier 1: Request validation
    VALIDATION_VOTE = "validation_vote"        # Tier 1: Submit validation vote


@dataclass
class Discovery:
    """
    Represents a validated scientific discovery or expression.
    
    This class encapsulates a discovered expression along with its evaluation
    metrics, provenance information, and validation history. It serves as the
    primary unit of knowledge in the shared knowledge base.
    
    Attributes:
        expression: The discovered mathematical/symbolic expression
        reward: Numerical reward/fitness score for the discovery
        timestamp: When the discovery was made
        discovered_by: ID of the agent that made the discovery
        version: Semantic version string (e.g., "v1.2.0")
        experiment_run_id: Unique identifier for the experiment run
        contributing_agents: List of agent IDs that contributed
        validation_score: Normalized score from validation process
        validation_votes: Dict mapping agent IDs to their votes
        metadata: Additional discovery-specific information
        refinement_count: Number of times this discovery was refined
    """
    expression: Expression  # Using the type alias for better type safety
    reward: float
    timestamp: datetime
    discovered_by: str
    version: str = "v1.0.0"
    experiment_run_id: Optional[str] = None
    contributing_agents: List[str] = field(default_factory=list)
    validation_score: Optional[float] = None
    validation_votes: Dict[str, bool] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    refinement_count: int = 0
    
    def __post_init__(self):
        """Ensure contributing_agents includes the discoverer."""
        if self.discovered_by and self.discovered_by not in self.contributing_agents:
            self.contributing_agents.insert(0, self.discovered_by)
    
    @property
    def is_validated(self) -> bool:
        """Check if the discovery has been validated."""
        return self.validation_score is not None and self.validation_score > 0
    
    @property
    def approval_rate(self) -> float:
        """Calculate the approval rate from validation votes."""
        if not self.validation_votes:
            return 0.0
        approvals = sum(1 for vote in self.validation_votes.values() if vote)
        return approvals / len(self.validation_votes)
    
    def increment_version(self, version_type: str = "minor") -> None:
        """
        Increment the semantic version.
        
        Args:
            version_type: Type of version increment ("major", "minor", "patch")
        """
        if not self.version.startswith("v"):
            self.version = "v" + self.version
            
        version_parts = self.version[1:].split(".")
        if len(version_parts) != 3:
            version_parts = ["1", "0", "0"]
        
        major, minor, patch = map(int, version_parts)
        
        if version_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif version_type == "minor":
            minor += 1
            patch = 0
        else:  # patch
            patch += 1
        
        self.version = f"v{major}.{minor}.{patch}"


@dataclass
class Message:
    """
    Communication packet for inter-agent messaging.
    
    Messages form the basis of the tiered communication protocol, enabling
    both strategic knowledge sharing (Tier 1) and tactical coordination (Tier 2).
    
    Attributes:
        msg_type: Type of message from MessageType enum
        sender_id: Unique identifier of the sending agent
        timestamp: When the message was created
        content: Message payload (type depends on msg_type)
        priority: Message priority (higher = more important)
        ttl: Time-to-live in seconds (None = no expiration)
        metadata: Additional message-specific information
    """
    msg_type: MessageType
    sender_id: str
    timestamp: datetime
    content: Any
    priority: int = 0
    ttl: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set default TTL based on message type if not specified."""
        if self.ttl is None:
            if self.msg_type == MessageType.TACTICAL_VECTOR:
                self.ttl = 60.0  # 60 seconds for tactical messages
            elif self.msg_type == MessageType.DISCOVERY_POINTER:
                self.ttl = 3600.0  # 1 hour for discovery pointers
            # Validation messages don't expire by default
    
    @property
    def is_expired(self) -> bool:
        """Check if the message has expired based on TTL."""
        if self.ttl is None:
            return False
        age = (datetime.now() - self.timestamp).total_seconds()
        return age > self.ttl
    
    @property
    def age(self) -> float:
        """Get the age of the message in seconds."""
        return (datetime.now() - self.timestamp).total_seconds()
    
    @property
    def remaining_ttl(self) -> Optional[float]:
        """Get remaining time-to-live in seconds."""
        if self.ttl is None:
            return None
        remaining = self.ttl - self.age
        return max(0.0, remaining)
    
    def is_tier1(self) -> bool:
        """Check if this is a Tier 1 (strategic) message."""
        tier1_types = {
            MessageType.DISCOVERY_POINTER,
            MessageType.VALIDATION_REQUEST,
            MessageType.VALIDATION_VOTE
        }
        return self.msg_type in tier1_types
    
    def is_tier2(self) -> bool:
        """Check if this is a Tier 2 (tactical) message."""
        return self.msg_type == MessageType.TACTICAL_VECTOR


# Example usage and testing
if __name__ == "__main__":
    # Create example discovery with type-safe expression
    discovery = Discovery(
        expression="x^2 + 2*x + 1",  # String expression
        reward=0.95,
        timestamp=datetime.now(),
        discovered_by="explorer_0",
        metadata={"complexity": 5, "mse": 0.001}
    )
    
    # Example with structured expression
    structured_discovery = Discovery(
        expression={"op": "add", "args": [{"op": "pow", "args": ["x", 2]}, {"op": "mul", "args": [2, "x"]}, 1]},
        reward=0.95,
        timestamp=datetime.now(),
        discovered_by="explorer_1",
        metadata={"representation": "tree"}
    )
    
    print(f"String expression: {discovery.expression}")
    print(f"Structured expression: {structured_discovery.expression}")
    print(f"Version: {discovery.version}")
    print(f"Is validated: {discovery.is_validated}")
    
    # Create example messages
    discovery_msg = Message(
        msg_type=MessageType.DISCOVERY_POINTER,
        sender_id="explorer_0",
        timestamp=datetime.now(),
        content={"discovery_id": "disc_001", "expression": discovery.expression},
        priority=2
    )
    
    tactical_msg = Message(
        msg_type=MessageType.TACTICAL_VECTOR,
        sender_id="refiner_1",
        timestamp=datetime.now(),
        content=[0.1, -0.5, 0.3, 0.8],  # Example latent vector
        priority=0
    )
    
    print(f"\nDiscovery message TTL: {discovery_msg.ttl}s")
    print(f"Tactical message TTL: {tactical_msg.ttl}s")
    print(f"Is Tier 1: {discovery_msg.is_tier1()}")
    print(f"Is Tier 2: {tactical_msg.is_tier2()}")