# JanusAI/scripts/examples/multi_agent/validator_agent_example.py
# JanusAI - Example Validator Agent Implementation
"""
Example Validator Agent Implementation
=====================================

This module demonstrates the recommended pattern for implementing a Validator agent
that follows the "LLM as Peer Reviewer" best practice from the documentation.

The validator prioritizes empirical evidence while optionally consulting an LLM
for ambiguous cases, treating it as an expert peer reviewer rather than an oracle.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from schemas import Discovery, MessageType, AgentRole
from knowledge import MessageBus, SharedKnowledgeBase

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Results from validating a discovery."""
    empirical_score: float
    complexity_score: float
    llm_similarity: Optional[float] = None
    final_vote: bool = False
    confidence: float = 0.0
    evidence: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.evidence is None:
            self.evidence = {}


class SmartValidatorAgent:
    """
    A sophisticated validator agent that implements the recommended validation pattern.
    
    This validator:
    1. Prioritizes empirical testing on held-out data
    2. Considers expression complexity (Occam's Razor)
    3. Optionally consults an LLM for ambiguous cases
    4. Makes weighted decisions based on multiple factors
    """
    
    def __init__(self, 
                 agent_id: str,
                 message_bus: MessageBus,
                 knowledge_base: SharedKnowledgeBase,
                 validation_data: np.ndarray,
                 llm_client: Optional[Any] = None,
                 empirical_weight: float = 0.6,
                 complexity_weight: float = 0.2,
                 llm_weight: float = 0.2):
        """
        Initialize the validator agent.
        
        Args:
            agent_id: Unique identifier for this agent
            message_bus: Communication bus for messages
            knowledge_base: Shared knowledge store
            validation_data: Held-out data for empirical validation
            llm_client: Optional LLM client for peer review
            empirical_weight: Weight for empirical fitness score
            complexity_weight: Weight for simplicity score
            llm_weight: Weight for LLM similarity (if used)
        """
        self.agent_id = agent_id
        self.role = AgentRole.VALIDATOR
        self.message_bus = message_bus
        self.knowledge_base = knowledge_base
        self.validation_data = validation_data
        self.llm_client = llm_client
        
        # Validation weights (should sum to 1.0)
        self.empirical_weight = empirical_weight
        self.complexity_weight = complexity_weight
        self.llm_weight = llm_weight
        
        # Thresholds
        self.approval_threshold = 0.7
        self.ambiguity_threshold = 0.1  # When to consult LLM
        
        logger.info(f"Initialized {agent_id} with weights: "
                   f"empirical={empirical_weight}, complexity={complexity_weight}, "
                   f"llm={llm_weight}")
    
    def process_messages(self) -> None:
        """Check for and process validation requests."""
        messages = self.message_bus.get_messages(self.agent_id, max_messages=10)
        
        for msg in messages:
            if msg.msg_type == MessageType.VALIDATION_REQUEST:
                try:
                    self._handle_validation_request(msg.content)
                except Exception as e:
                    logger.error(f"Error validating discovery: {e}")
    
    def _handle_validation_request(self, request_content: Dict[str, Any]) -> None:
        """Handle a validation request with the full validation pipeline."""
        discovery_id = request_content['discovery_id']
        discovery = request_content['discovery']
        
        logger.info(f"Validating discovery {discovery_id}: {discovery.expression}")
        
        # Perform comprehensive validation
        validation_result = self._validate_discovery(discovery)
        
        # Submit vote with detailed evidence
        self.knowledge_base.vote_on_discovery(
            agent_id=self.agent_id,
            discovery_id=discovery_id,
            approve=validation_result.final_vote,
            evidence=validation_result.evidence
        )
        
        logger.info(f"Voted {'APPROVE' if validation_result.final_vote else 'REJECT'} "
                   f"on {discovery_id} with confidence {validation_result.confidence:.3f}")
    
    def _validate_discovery(self, discovery: Discovery) -> ValidationResult:
        """
        Comprehensive validation following the recommended pattern.
        
        Steps:
        1. Empirical evaluation on held-out data
        2. Complexity assessment (Occam's Razor)
        3. Optional LLM consultation for ambiguous cases
        4. Weighted decision making
        """
        result = ValidationResult(
            empirical_score=0.0,
            complexity_score=0.0,
            evidence={}
        )
        
        # Step 1: Empirical evaluation (primary factor)
        empirical_score, mse = self._evaluate_empirically(discovery.expression)
        result.empirical_score = empirical_score
        result.evidence['mse'] = mse
        result.evidence['empirical_score'] = empirical_score
        
        # Step 2: Complexity assessment
        complexity_score = self._assess_complexity(discovery.expression)
        result.complexity_score = complexity_score
        result.evidence['complexity_score'] = complexity_score
        result.evidence['expression_length'] = len(str(discovery.expression))
        
        # Step 3: Check if we need LLM consultation
        # We consult LLM if the empirical score is in the ambiguous range
        if abs(empirical_score - 0.5) < self.ambiguity_threshold and self.llm_client:
            llm_similarity = self._consult_llm_peer(discovery)
            result.llm_similarity = llm_similarity
            result.evidence['llm_consulted'] = True
            result.evidence['llm_similarity'] = llm_similarity
        else:
            # Adjust weights if LLM not used
            total_weight = self.empirical_weight + self.complexity_weight
            empirical_weight = self.empirical_weight / total_weight
            complexity_weight = self.complexity_weight / total_weight
            llm_weight = 0.0
        
        # Step 4: Calculate weighted score
        if result.llm_similarity is not None:
            weighted_score = (
                self.empirical_weight * result.empirical_score +
                self.complexity_weight * result.complexity_score +
                self.llm_weight * result.llm_similarity
            )
        else:
            weighted_score = (
                empirical_weight * result.empirical_score +
                complexity_weight * result.complexity_score
            )
        
        # Make final decision
        result.final_vote = weighted_score >= self.approval_threshold
        result.confidence = abs(weighted_score - self.approval_threshold)
        result.evidence['weighted_score'] = weighted_score
        result.evidence['decision_threshold'] = self.approval_threshold
        
        return result
    
    def _evaluate_empirically(self, expression: Any) -> Tuple[float, float]:
        """
        Evaluate the expression on held-out validation data.
        
        Returns:
            Tuple of (normalized_score, mean_squared_error)
        """
        try:
            # This is a simplified example - in practice, you'd evaluate
            # the expression on your validation data
            # For now, simulate with random performance
            mse = np.random.uniform(0.001, 0.1)
            
            # Convert MSE to normalized score (0-1, higher is better)
            # Using exponential decay function
            normalized_score = np.exp(-mse * 10)
            
            return normalized_score, mse
            
        except Exception as e:
            logger.error(f"Empirical evaluation failed: {e}")
            return 0.0, float('inf')
    
    def _assess_complexity(self, expression: Any) -> float:
        """
        Assess expression complexity (Occam's Razor).
        
        Returns:
            Complexity score (0-1, higher means simpler/better)
        """
        expr_str = str(expression)
        
        # Simple heuristics for complexity
        # In practice, use proper expression tree analysis
        length_penalty = len(expr_str) / 100.0  # Normalize by expected max length
        operator_count = expr_str.count('+') + expr_str.count('*') + expr_str.count('^')
        operator_penalty = operator_count / 10.0
        
        # Convert to score (higher is simpler/better)
        complexity_score = 1.0 / (1.0 + length_penalty + operator_penalty)
        
        return min(max(complexity_score, 0.0), 1.0)
    
    def _consult_llm_peer(self, discovery: Discovery) -> float:
        """
        Consult LLM as a peer reviewer for ambiguous cases.
        
        Returns:
            Similarity score between discovery and LLM suggestion (0-1)
        """
        if not self.llm_client:
            return 0.5  # Neutral if no LLM available
        
        try:
            # Example prompt for LLM consultation
            prompt = f"""
            As an expert in symbolic regression, please review this discovered expression:
            Expression: {discovery.expression}
            Reward: {discovery.reward}
            
            Is this a reasonable mathematical expression for the given performance?
            Suggest a similar or better expression if you see improvements.
            """
            
            # In practice, you'd call your LLM here
            # llm_response = self.llm_client.query(prompt)
            
            # For now, simulate with random similarity
            similarity = np.random.uniform(0.3, 0.9)
            
            logger.info(f"LLM peer review similarity: {similarity:.3f}")
            return similarity
            
        except Exception as e:
            logger.error(f"LLM consultation failed: {e}")
            return 0.5  # Neutral on failure


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create infrastructure
    message_bus = MessageBus()
    knowledge_base = SharedKnowledgeBase(validation_threshold=2, message_bus=message_bus)
    
    # Create validation data (held-out dataset)
    validation_data = np.random.randn(100, 3)  # 100 samples, 3 features
    
    # Create validator
    validator = SmartValidatorAgent(
        agent_id="smart_validator_001",
        message_bus=message_bus,
        knowledge_base=knowledge_base,
        validation_data=validation_data,
        llm_client=None,  # Would pass actual LLM client here
        empirical_weight=0.6,  # Prioritize empirical evidence
        complexity_weight=0.2,
        llm_weight=0.2
    )
    
    # Simulate a discovery to validate
    from datetime import datetime
    
    discovery = Discovery(
        expression="x^2 + 2*x + 1",
        reward=0.95,
        timestamp=datetime.now(),
        discovered_by="explorer_001"
    )
    
    # Propose it
    discovery_id = knowledge_base.propose_discovery("explorer_001", discovery)
    
    # Validator processes messages
    validator.process_messages()
    
    # Check the evidence
    if discovery_id in knowledge_base.pending_validations:
        votes = knowledge_base.pending_validations[discovery_id]['votes']
        for agent, (vote, evidence) in votes.items():
            print(f"\n{agent} voted: {'APPROVE' if vote else 'REJECT'}")
            print(f"Evidence: {evidence}")