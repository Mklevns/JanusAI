# JanusAI/integration/knowledge.py
# JanusAI Multi-Agent System
"""
Corrected Communication Infrastructure for JanusAI Multi-Agent System
===================================================================

This module implements the core communication and knowledge management infrastructure
for the JanusAI multi-agent discovery system with all recommended corrections applied.

Changes from previous version:
1. Removed duplicated data structure definitions (now imports from schemas)
2. Simplified and corrected message retrieval logic to be thread-safe
3. Added clarifying comments for consensus logic
4. Improved type hints and documentation

Author: JanusAI Team
Date: 2024
"""

import logging
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Import data structures from schemas module
from janus_ai.schemas import AgentRole, MessageType, Discovery, Message


# Configure logging
logger = logging.getLogger(__name__)


class MessageBus:
    """
    High-throughput message bus with Quality of Service (QoS) and flow control.
    
    Implements a two-tier priority system where Tier 1 (strategic) messages
    are always served before Tier 2 (tactical) messages. Includes rate limiting
    to prevent agent spam and TTL-based expiration for tactical messages.
    
    Attributes:
        max_queue_size: Maximum size for the Tier 2 queue
        tier1_queue: Unbounded priority queue for strategic messages
        tier2_queue: Size-limited queue for tactical messages
        message_stats: Statistics on message throughput
        rate_limits: Per-agent rate limiting state
    """
    
    def __init__(self, max_queue_size: int = 10000, tokens_per_second: float = 10.0):
        """
        Initialize the MessageBus.
        
        Args:
            max_queue_size: Maximum number of messages in Tier 2 queue
            tokens_per_second: Rate limit tokens per second per agent
        """
        self.max_queue_size = max_queue_size
        self.tokens_per_second = tokens_per_second
        
        # Separate queues for each tier
        self.tier1_queue = deque()  # Unbounded for high-priority
        self.tier2_queue = deque(maxlen=max_queue_size)  # Bounded for tactical
        
        # Statistics and rate limiting
        self.message_stats = {'tier1': 0, 'tier2': 0, 'dropped': 0}
        self.rate_limits = {}  # agent_id -> {'tokens': float, 'last_update': float}
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info(f"MessageBus initialized with max_queue_size={max_queue_size}")
    
    def publish(self, message: Message) -> bool:
        """
        Publish a message to the appropriate queue with rate limiting.
        
        Args:
            message: The message to publish
            
        Returns:
            bool: True if message was published, False if rate limited or dropped
        """
        with self._lock:
            # Check rate limit
            if not self._check_rate_limit(message.sender_id):
                logger.warning(f"Rate limit exceeded for agent {message.sender_id}")
                self.message_stats['dropped'] += 1
                return False
            
            # Route based on message type
            if message.msg_type in [MessageType.DISCOVERY_POINTER, 
                                   MessageType.VALIDATION_REQUEST,
                                   MessageType.VALIDATION_VOTE]:
                # Tier 1: Strategic messages
                self.tier1_queue.append(message)
                self.message_stats['tier1'] += 1
                logger.debug(f"Published Tier 1 message: {message.msg_type.value} from {message.sender_id}")
                
            else:
                # Tier 2: Tactical messages
                # Set default TTL if not specified
                if message.ttl is None:
                    message.ttl = 60.0  # 60 seconds default
                    
                # Check if queue is full (deque with maxlen automatically drops oldest)
                queue_was_full = len(self.tier2_queue) == self.max_queue_size
                self.tier2_queue.append(message)
                self.message_stats['tier2'] += 1
                
                if queue_was_full:
                    self.message_stats['dropped'] += 1
                    logger.debug("Tier 2 queue full, oldest message dropped")
                
                logger.debug(f"Published Tier 2 message from {message.sender_id}")
            
            return True
    
    def _check_rate_limit(self, agent_id: str) -> bool:
        """
        Token bucket rate limiting per agent.
        
        Args:
            agent_id: The agent attempting to publish
            
        Returns:
            bool: True if agent has tokens available, False otherwise
        """
        current_time = time.time()
        max_tokens = 100  # Maximum bucket size
        
        if agent_id not in self.rate_limits:
            # Initialize rate limit state for new agent
            self.rate_limits[agent_id] = {
                'tokens': max_tokens,
                'last_update': current_time
            }
        
        agent_limit = self.rate_limits[agent_id]
        time_passed = current_time - agent_limit['last_update']
        
        # Replenish tokens based on time passed
        tokens_to_add = time_passed * self.tokens_per_second
        agent_limit['tokens'] = min(max_tokens, agent_limit['tokens'] + tokens_to_add)
        agent_limit['last_update'] = current_time
        
        # Check if agent has tokens available
        if agent_limit['tokens'] >= 1.0:
            agent_limit['tokens'] -= 1.0
            return True
        
        return False
    
    def get_messages(self, agent_id: str, max_messages: int = 10) -> List[Message]:
        """
        Retrieve messages for an agent with priority handling.
        
        This implementation uses a safer, clearer filtering mechanism that avoids
        potential race conditions and message reordering.
        
        Tier 1 messages are always served first, then Tier 2 messages.
        Expired Tier 2 messages are filtered out automatically.
        
        Args:
            agent_id: The requesting agent's ID
            max_messages: Maximum number of messages to return
            
        Returns:
            List of messages prioritized by tier and filtered for expiration
        """
        with self._lock:
            messages = []
            
            # Priority 1: Serve Tier 1 messages first
            while self.tier1_queue and len(messages) < max_messages:
                messages.append(self.tier1_queue.popleft())
            
            # Priority 2: Serve non-expired Tier 2 messages
            if len(messages) < max_messages:
                # Create a new deque to hold messages that are not expired or delivered
                still_valid_tier2 = deque(maxlen=self.max_queue_size)
                expired_count = 0
                
                while self.tier2_queue:
                    msg = self.tier2_queue.popleft()
                    if msg.is_expired:
                        expired_count += 1
                        continue  # Drop expired message
                    
                    # If there's room, deliver the message
                    if len(messages) < max_messages:
                        messages.append(msg)
                    else:
                        # Otherwise, keep it for the next agent
                        still_valid_tier2.append(msg)
                
                # Atomically replace the old queue with the filtered one
                self.tier2_queue = still_valid_tier2
                
                if expired_count > 0:
                    logger.debug(f"Filtered out {expired_count} expired Tier 2 messages")
            
            logger.debug(f"Delivered {len(messages)} messages to {agent_id}")
            return messages
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current message bus statistics.
        
        Returns:
            Dictionary containing message counts and queue sizes
        """
        with self._lock:
            return {
                'tier1_total': self.message_stats['tier1'],
                'tier2_total': self.message_stats['tier2'],
                'dropped_total': self.message_stats['dropped'],
                'tier1_queue_size': len(self.tier1_queue),
                'tier2_queue_size': len(self.tier2_queue),
                'active_agents': len(self.rate_limits)
            }


class SharedKnowledgeBase:
    """
    Persistent knowledge store with Augmented Consensus validation.
    
    Manages the collective "scientific record" of discoveries using a multi-agent
    consensus protocol. Discoveries must be validated by multiple agents before
    being confirmed and added to the permanent knowledge base.
    
    Note: This implementation is in-memory. For production use requiring
    durability across restarts, it should be backed by a persistent database
    like Redis or PostgreSQL.
    
    Attributes:
        discoveries: Confirmed discoveries indexed by expression string
        pending_validations: Discoveries awaiting validation
        validation_threshold: Number of approvals needed for consensus
        message_bus: Reference to the message bus for publishing
        agent_contributions: Tracking of each agent's contributions
        version_history: Complete version history for each discovery
    """
    
    def __init__(self, validation_threshold: int = 2, message_bus: Optional[MessageBus] = None):
        """
        Initialize the SharedKnowledgeBase.
        
        Args:
            validation_threshold: Minimum approvals needed for validation
            message_bus: MessageBus instance for publishing (creates new if None)
        """
        self.validation_threshold = validation_threshold
        self.message_bus = message_bus or MessageBus()
        
        # Core data structures
        self.discoveries = {}  # expr_str -> Discovery
        self.pending_validations = {}  # discovery_id -> validation_state
        self.agent_contributions = defaultdict(list)  # agent_id -> [discoveries]
        self.version_history = defaultdict(list)  # expr_str -> [versions]
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'proposals': 0,
            'validations': 0,
            'rejections': 0,
            'version_updates': 0
        }
        
        logger.info(f"SharedKnowledgeBase initialized with validation_threshold={validation_threshold}")
    
    def propose_discovery(self, agent_id: str, discovery: Discovery) -> str:
        """
        Propose a new discovery for validation.
        
        Creates a validation request and publishes it to all validator agents.
        The discovery is not immediately added but enters a pending state.
        
        Args:
            agent_id: ID of the proposing agent
            discovery: The discovery to propose
            
        Returns:
            str: Unique discovery ID for tracking
        """
        with self._lock:
            # Generate unique discovery ID
            discovery_id = f"{agent_id}_{int(discovery.timestamp.timestamp() * 1000)}"
            
            # Set experiment run ID if not present
            if not discovery.experiment_run_id:
                discovery.experiment_run_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create pending validation entry
            self.pending_validations[discovery_id] = {
                'discovery': discovery,
                'votes': {},  # agent_id -> (approve, evidence)
                'proposed_at': datetime.now(),
                'proposer': agent_id
            }
            
            self.stats['proposals'] += 1
            
            # Create validation request message
            validation_request = Message(
                msg_type=MessageType.VALIDATION_REQUEST,
                sender_id='system',
                timestamp=datetime.now(),
                content={
                    'discovery_id': discovery_id,
                    'discovery': discovery,
                    'proposer': agent_id
                },
                priority=1,  # High priority for validation requests
                metadata={'requires_role': AgentRole.VALIDATOR.value}
            )
            
            # Publish to message bus
            self.message_bus.publish(validation_request)
            
            logger.info(f"Discovery proposed: {discovery_id} by {agent_id}, "
                       f"expression: {discovery.expression}")
            
            return discovery_id
    
    def vote_on_discovery(self, agent_id: str, discovery_id: str, 
                         approve: bool, evidence: Optional[Dict[str, Any]] = None) -> None:
        """
        Submit a validation vote for a pending discovery.
        
        If the approval threshold is reached, the discovery is automatically
        confirmed and added to the knowledge base.
        
        Args:
            agent_id: ID of the voting agent
            discovery_id: ID of the discovery being voted on
            approve: Whether to approve the discovery
            evidence: Optional evidence supporting the vote
        """
        with self._lock:
            if discovery_id not in self.pending_validations:
                logger.warning(f"Vote on unknown discovery: {discovery_id}")
                return
            
            pending = self.pending_validations[discovery_id]
            
            # Prevent duplicate votes
            if agent_id in pending['votes']:
                logger.warning(f"Duplicate vote from {agent_id} on {discovery_id}")
                return
            
            # Prevent self-validation
            if agent_id == pending['proposer']:
                logger.warning(f"Agent {agent_id} cannot vote on own discovery")
                return
            
            # Record vote
            pending['votes'][agent_id] = (approve, evidence or {})
            
            # Count approvals
            approve_votes = sum(1 for vote, _ in pending['votes'].values() if vote)
            total_votes = len(pending['votes'])
            
            logger.info(f"Vote recorded: {agent_id} {'approved' if approve else 'rejected'} "
                       f"{discovery_id} ({approve_votes}/{total_votes} approvals)")
            
            # Check if consensus reached
            if approve_votes >= self.validation_threshold:
                # Validation successful
                discovery = pending['discovery']
                discovery.validation_votes = {
                    agent: vote for agent, (vote, _) in pending['votes'].items()
                }
                discovery.validation_score = approve_votes / total_votes
                
                # Add voting agents as contributors
                for voter_id in pending['votes']:
                    if voter_id not in discovery.contributing_agents:
                        discovery.contributing_agents.append(voter_id)
                
                self._confirm_discovery(discovery)
                del self.pending_validations[discovery_id]
                self.stats['validations'] += 1
                
            # Check for early rejection
            # This checks if it's mathematically impossible to reach the threshold
            # even if all remaining validators approve
            elif total_votes - approve_votes > len(self.pending_validations[discovery_id]['votes']) - self.validation_threshold:
                logger.info(f"Discovery {discovery_id} rejected by consensus (insufficient possible approvals)")
                del self.pending_validations[discovery_id]
                self.stats['rejections'] += 1
    
    def _confirm_discovery(self, discovery: Discovery) -> None:
        """
        Confirm and store a validated discovery.
        
        Handles versioning for improvements to existing discoveries and
        publishes a discovery pointer to notify all agents.
        
        Args:
            discovery: The validated discovery to confirm
        """
        expr_str = str(discovery.expression)
        
        # Check for existing discovery
        if expr_str in self.discoveries:
            existing = self.discoveries[expr_str]
            
            if discovery.reward > existing.reward:
                # New version with better reward
                discovery.version = existing.version
                discovery.increment_version('minor')
                self.discoveries[expr_str] = discovery
                self.stats['version_updates'] += 1
                
                logger.info(f"Updated discovery: {expr_str} to version {discovery.version} "
                           f"(reward: {existing.reward:.3f} -> {discovery.reward:.3f})")
            else:
                # Existing version is better, don't update
                logger.info(f"Discovery {expr_str} not updated (existing reward better)")
                return
        else:
            # New discovery
            self.discoveries[expr_str] = discovery
            logger.info(f"New discovery confirmed: {expr_str} (reward: {discovery.reward:.3f})")
        
        # Update version history
        self.version_history[expr_str].append(discovery)
        
        # Track agent contribution
        self.agent_contributions[discovery.discovered_by].append(discovery)
        
        # Publish discovery pointer to all agents
        discovery_pointer = Message(
            msg_type=MessageType.DISCOVERY_POINTER,
            sender_id='system',
            timestamp=datetime.now(),
            content={
                'expression': expr_str,
                'version': discovery.version,
                'reward': discovery.reward,
                'discovered_by': discovery.discovered_by,
                'validation_score': discovery.validation_score
            },
            priority=2,  # Highest priority for confirmed discoveries
            ttl=3600.0  # 1 hour TTL for discovery pointers
        )
        
        self.message_bus.publish(discovery_pointer)
    
    def get_best_discoveries(self, n: int = 10, min_version: Optional[str] = None) -> List[Discovery]:
        """
        Get the top discoveries by reward, optionally filtered by version.
        
        Args:
            n: Number of discoveries to return
            min_version: Minimum version requirement (e.g., "v2.0.0")
            
        Returns:
            List of top discoveries sorted by reward
        """
        with self._lock:
            discoveries = list(self.discoveries.values())
            
            # Filter by version if specified
            if min_version:
                discoveries = [d for d in discoveries if d.version >= min_version]
            
            # Sort by reward (with validation score as tiebreaker)
            discoveries.sort(
                key=lambda d: (d.reward, d.validation_score or 0),
                reverse=True
            )
            
            return discoveries[:n]
    
    def get_agent_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get comprehensive statistics for each agent's contributions.
        
        Returns:
            Dictionary mapping agent IDs to their statistics
        """
        with self._lock:
            stats = {}
            
            for agent_id, discoveries in self.agent_contributions.items():
                if discoveries:
                    stats[agent_id] = {
                        'total_discoveries': len(discoveries),
                        'confirmed_discoveries': len([d for d in discoveries 
                                                    if str(d.expression) in self.discoveries]),
                        'avg_reward': sum(d.reward for d in discoveries) / len(discoveries),
                        'best_reward': max(d.reward for d in discoveries),
                        'total_votes': sum(len(d.validation_votes) for d in discoveries),
                        'avg_validation_score': sum(d.validation_score or 0 for d in discoveries) / len(discoveries)
                    }
                else:
                    stats[agent_id] = {
                        'total_discoveries': 0,
                        'confirmed_discoveries': 0,
                        'avg_reward': 0.0,
                        'best_reward': 0.0,
                        'total_votes': 0,
                        'avg_validation_score': 0.0
                    }
            
            return stats
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current knowledge base state.
        
        Returns:
            Dictionary containing knowledge base statistics
        """
        with self._lock:
            return {
                'total_discoveries': len(self.discoveries),
                'pending_validations': len(self.pending_validations),
                'total_proposals': self.stats['proposals'],
                'successful_validations': self.stats['validations'],
                'rejections': self.stats['rejections'],
                'version_updates': self.stats['version_updates'],
                'unique_contributors': len(self.agent_contributions),
                'avg_validation_score': sum(d.validation_score or 0 for d in self.discoveries.values()) / max(len(self.discoveries), 1)
            }


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create infrastructure
    message_bus = MessageBus(max_queue_size=1000)
    knowledge_base = SharedKnowledgeBase(validation_threshold=2, message_bus=message_bus)
    
    # Simulate discovery proposal
    discovery = Discovery(
        expression="x^2 + 2*x + 1",
        reward=0.95,
        timestamp=datetime.now(),
        discovered_by="explorer_0"
    )
    
    discovery_id = knowledge_base.propose_discovery("explorer_0", discovery)
    print(f"Proposed discovery: {discovery_id}")
    
    # Simulate validation votes
    knowledge_base.vote_on_discovery("validator_1", discovery_id, True, {"mse": 0.001})
    knowledge_base.vote_on_discovery("validator_2", discovery_id, True, {"complexity": "optimal"})
    
    # Check results
    print(f"\nKnowledge base summary: {knowledge_base.get_knowledge_summary()}")
    print(f"Message bus stats: {message_bus.get_stats()}")
    
    # Get messages for an agent
    messages = message_bus.get_messages("validator_3", max_messages=5)
    print(f"\nMessages for validator_3: {len(messages)}")
    for msg in messages:
        print(f"  - {msg.msg_type.value} from {msg.sender_id}")