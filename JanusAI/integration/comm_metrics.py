# JanusAI/integration/comm_metrics.py
"""
Communication Metrics and Reward Functions
==========================================

This module provides comprehensive metrics and reward functions for
evaluating and optimizing emergent communication protocols.

Metrics include:
- Bandwidth efficiency
- Reconstruction accuracy
- Symbol complexity and reuse
- Communication latency
- Task success correlation

Author: JanusAI Team
Date: 2024
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CommunicationMetrics:
    """Container for communication performance metrics."""
    # Efficiency metrics
    avg_message_length: float = 0.0
    bandwidth_usage: float = 0.0  # bits per message
    compression_ratio: float = 0.0
    
    # Accuracy metrics
    reconstruction_accuracy: float = 0.0
    semantic_similarity: float = 0.0
    task_success_rate: float = 0.0
    
    # Complexity metrics
    vocabulary_entropy: float = 0.0
    bigram_entropy: float = 0.0
    symbol_reuse_rate: float = 0.0
    emergent_patterns: Dict[str, int] = field(default_factory=dict)
    
    # Temporal metrics
    avg_latency: float = 0.0  # steps from discovery to validation
    convergence_time: float = 0.0  # steps to consensus
    
    # Correlation metrics
    comm_task_correlation: float = 0.0  # correlation between comm quality and task success


class CommunicationAnalyzer:
    """Analyze communication patterns and compute metrics."""
    
    def __init__(self, vocab_size: int, max_message_length: int):
        self.vocab_size = vocab_size
        self.max_message_length = max_message_length
        
        # Track communication history
        self.message_history = []
        self.reconstruction_errors = []
        self.task_outcomes = []
        self.latencies = []
        
        # Symbol tracking
        self.symbol_counts = Counter()
        self.bigram_counts = Counter()
        self.pattern_counts = Counter()
        
        # Temporal tracking
        self.discovery_times = {}
        self.validation_times = {}
    
    def record_message(self, message: torch.Tensor, 
                      reconstruction_error: float,
                      task_success: bool,
                      timestamp: datetime = None):
        """Record a communication instance."""
        timestamp = timestamp or datetime.now()
        
        # Convert message to token indices
        if message.dim() == 3:  # One-hot encoding
            tokens = message.argmax(dim=-1).cpu().numpy()
        else:  # Already indices
            tokens = message.cpu().numpy()
        
        # Flatten batch dimension if present
        if tokens.ndim > 1:
            tokens = tokens.flatten()
        
        # Record basic metrics
        self.message_history.append(tokens)
        self.reconstruction_errors.append(reconstruction_error)
        self.task_outcomes.append(task_success)
        
        # Update symbol counts
        for token in tokens:
            if token < self.vocab_size:  # Valid token
                self.symbol_counts[token] += 1
        
        # Update bigram counts
        for i in range(len(tokens) - 1):
            bigram = (tokens[i], tokens[i + 1])
            self.bigram_counts[bigram] += 1
        
        # Look for recurring patterns (3-grams and 4-grams)
        for pattern_len in [3, 4]:
            for i in range(len(tokens) - pattern_len + 1):
                pattern = tuple(tokens[i:i + pattern_len])
                self.pattern_counts[pattern] += 1
    
    def record_discovery(self, discovery_id: str, timestamp: datetime = None):
        """Record when a discovery was made."""
        self.discovery_times[discovery_id] = timestamp or datetime.now()
    
    def record_validation(self, discovery_id: str, timestamp: datetime = None):
        """Record when a discovery was validated."""
        self.validation_times[discovery_id] = timestamp or datetime.now()
        
        # Calculate latency
        if discovery_id in self.discovery_times:
            latency = (self.validation_times[discovery_id] - 
                      self.discovery_times[discovery_id]).total_seconds()
            self.latencies.append(latency)
    
    def compute_metrics(self) -> CommunicationMetrics:
        """Compute comprehensive communication metrics."""
        metrics = CommunicationMetrics()
        
        if not self.message_history:
            return metrics
        
        # Efficiency metrics
        metrics.avg_message_length = self._compute_avg_message_length()
        metrics.bandwidth_usage = self._compute_bandwidth()
        metrics.compression_ratio = self._compute_compression_ratio()
        
        # Accuracy metrics
        metrics.reconstruction_accuracy = 1.0 - np.mean(self.reconstruction_errors)
        metrics.task_success_rate = np.mean(self.task_outcomes)
        
        # Complexity metrics
        metrics.vocabulary_entropy = self._compute_entropy(self.symbol_counts)
        metrics.bigram_entropy = self._compute_entropy(self.bigram_counts)
        metrics.symbol_reuse_rate = self._compute_reuse_rate()
        metrics.emergent_patterns = self._find_emergent_patterns()
        
        # Temporal metrics
        if self.latencies:
            metrics.avg_latency = np.mean(self.latencies)
        
        # Correlation metrics
        metrics.comm_task_correlation = self._compute_correlation()
        
        return metrics
    
    def _compute_avg_message_length(self) -> float:
        """Compute average effective message length."""
        lengths = []
        for msg in self.message_history:
            # Count non-padding tokens (assuming 0 is padding)
            effective_length = np.sum(msg != 0)
            lengths.append(effective_length)
        return np.mean(lengths) if lengths else 0
    
    def _compute_bandwidth(self) -> float:
        """Compute bits per message."""
        if not self.message_history:
            return 0
        
        # Calculate entropy per position
        position_entropies = []
        for pos in range(self.max_message_length):
            pos_symbols = [msg[pos] if pos < len(msg) else 0 
                          for msg in self.message_history]
            pos_counts = Counter(pos_symbols)
            entropy = self._compute_entropy(pos_counts)
            position_entropies.append(entropy)
        
        # Total bandwidth is sum of position entropies
        return sum(position_entropies)
    
    def _compute_compression_ratio(self) -> float:
        """Compute compression ratio vs uniform encoding."""
        actual_bandwidth = self._compute_bandwidth()
        max_bandwidth = self.max_message_length * np.log2(self.vocab_size)
        return 1 - (actual_bandwidth / max_bandwidth) if max_bandwidth > 0 else 0
    
    def _compute_entropy(self, counts: Counter) -> float:
        """Compute Shannon entropy from count distribution."""
        total = sum(counts.values())
        if total == 0:
            return 0
        
        entropy = 0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _compute_reuse_rate(self) -> float:
        """Compute rate of symbol reuse vs unique symbols."""
        total_symbols = sum(self.symbol_counts.values())
        unique_symbols = len(self.symbol_counts)
        
        if unique_symbols == 0:
            return 0
        
        return total_symbols / unique_symbols
    
    def _find_emergent_patterns(self, min_count: int = 5) -> Dict[str, int]:
        """Find frequently recurring patterns."""
        emergent = {}
        
        # Find patterns that appear frequently
        for pattern, count in self.pattern_counts.items():
            if count >= min_count:
                pattern_str = "-".join(map(str, pattern))
                emergent[pattern_str] = count
        
        # Sort by frequency
        return dict(sorted(emergent.items(), key=lambda x: x[1], reverse=True)[:10])
    
    def _compute_correlation(self) -> float:
        """Compute correlation between communication quality and task success."""
        if len(self.reconstruction_errors) < 2:
            return 0
        
        # Convert to numpy arrays
        comm_quality = 1 - np.array(self.reconstruction_errors)
        task_success = np.array(self.task_outcomes, dtype=float)
        
        # Compute Pearson correlation
        return np.corrcoef(comm_quality, task_success)[0, 1]


class CommunicationReward:
    """
    Compute rewards for communication quality.
    
    Balances multiple objectives:
    - Reconstruction fidelity
    - Message efficiency
    - Task performance
    - Emergent structure
    """
    
    def __init__(self, 
                 fidelity_weight: float = 0.5,
                 efficiency_weight: float = 0.2,
                 task_weight: float = 0.3):
        self.fidelity_weight = fidelity_weight
        self.efficiency_weight = efficiency_weight
        self.task_weight = task_weight
    
    def compute_reward(self,
                      original_state: torch.Tensor,
                      reconstructed_state: torch.Tensor,
                      message: torch.Tensor,
                      task_success: bool,
                      max_message_length: int) -> Tuple[float, Dict[str, float]]:
        """
        Compute communication reward.
        
        Returns:
            Total reward and component breakdown
        """
        components = {}
        
        # Fidelity reward (negative MSE)
        mse = F.mse_loss(reconstructed_state, original_state)
        components['fidelity'] = -mse.item()
        
        # Efficiency reward (shorter messages are better)
        if message.dim() == 3:  # One-hot
            message_length = (message.sum(dim=-1) > 0).sum(dim=-1).float().mean()
        else:  # Indices
            message_length = (message != 0).sum(dim=-1).float().mean()
        
        components['efficiency'] = 1 - (message_length / max_message_length).item()
        
        # Task reward
        components['task'] = float(task_success)
        
        # Weighted sum
        total_reward = (
            self.fidelity_weight * components['fidelity'] +
            self.efficiency_weight * components['efficiency'] +
            self.task_weight * components['task']
        )
        
        return total_reward, components


class SymbolicLanguageAnalyzer:
    """
    Analyze emergent symbolic language properties.
    
    Looks for:
    - Compositional structure
    - Semantic clustering
    - Syntactic patterns
    """
    
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.expression_to_message = defaultdict(list)
        self.message_to_expression = defaultdict(list)
    
    def record_mapping(self, expression: str, message: torch.Tensor):
        """Record expression-message mapping."""
        # Convert to string representation
        if message.dim() == 3:
            tokens = message.argmax(dim=-1).cpu().numpy()
        else:
            tokens = message.cpu().numpy()
        
        msg_str = "-".join(map(str, tokens.flatten()))
        
        self.expression_to_message[expression].append(msg_str)
        self.message_to_expression[msg_str].append(expression)
    
    def analyze_compositionality(self) -> Dict[str, Any]:
        """Analyze if language shows compositional structure."""
        results = {
            'consistency': self._compute_consistency(),
            'semantic_clusters': self._find_semantic_clusters(),
            'syntactic_patterns': self._find_syntactic_patterns()
        }
        
        return results
    
    def _compute_consistency(self) -> float:
        """Measure how consistently expressions map to messages."""
        consistencies = []
        
        for expr, messages in self.expression_to_message.items():
            if len(messages) > 1:
                # Count most common message
                msg_counts = Counter(messages)
                most_common_count = msg_counts.most_common(1)[0][1]
                consistency = most_common_count / len(messages)
                consistencies.append(consistency)
        
        return np.mean(consistencies) if consistencies else 0
    
    def _find_semantic_clusters(self) -> Dict[str, List[str]]:
        """Find expressions that map to similar messages."""
        clusters = defaultdict(list)
        
        # Group expressions by message patterns
        for msg, expressions in self.message_to_expression.items():
            if len(expressions) > 1:
                clusters[msg] = expressions
        
        # Return top clusters
        return dict(sorted(clusters.items(), 
                          key=lambda x: len(x[1]), 
                          reverse=True)[:5])
    
    def _find_syntactic_patterns(self) -> Dict[str, int]:
        """Find common syntactic patterns in messages."""
        patterns = defaultdict(int)
        
        for messages in self.expression_to_message.values():
            for msg in messages:
                tokens = msg.split('-')
                
                # Look for patterns like "X-X" (repetition)
                if len(tokens) >= 2:
                    if tokens[0] == tokens[1]:
                        patterns['repetition'] += 1
                
                # Look for patterns like "A-B-A" (ABA structure)
                if len(tokens) >= 3:
                    if tokens[0] == tokens[2] and tokens[0] != tokens[1]:
                        patterns['ABA'] += 1
                
                # Look for ascending sequences
                if len(tokens) >= 3:
                    try:
                        int_tokens = [int(t) for t in tokens[:3]]
                        if int_tokens[1] == int_tokens[0] + 1 and int_tokens[2] == int_tokens[1] + 1:
                            patterns['ascending'] += 1
                    except:
                        pass
        
        return dict(patterns)


# Example visualization function
def visualize_communication_metrics(analyzer: CommunicationAnalyzer, 
                                  save_path: Optional[str] = None):
    """Create visualization of communication metrics."""
    import matplotlib.pyplot as plt
    
    metrics = analyzer.compute_metrics()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Vocabulary usage
    ax = axes[0, 0]
    vocab_usage = list(analyzer.symbol_counts.values())
    ax.hist(vocab_usage, bins=20, alpha=0.7)
    ax.set_title(f'Symbol Usage Distribution (Entropy: {metrics.vocabulary_entropy:.2f})')
    ax.set_xlabel('Usage Count')
    ax.set_ylabel('Number of Symbols')
    
    # Message length distribution
    ax = axes[0, 1]
    lengths = [len(msg) for msg in analyzer.message_history]
    ax.hist(lengths, bins=range(0, analyzer.max_message_length + 2), alpha=0.7)
    ax.set_title(f'Message Length Distribution (Avg: {metrics.avg_message_length:.2f})')
    ax.set_xlabel('Message Length')
    ax.set_ylabel('Frequency')
    
    # Task success vs reconstruction accuracy
    ax = axes[1, 0]
    if len(analyzer.reconstruction_errors) > 10:
        window = 10
        rec_acc = [1 - np.mean(analyzer.reconstruction_errors[i:i+window]) 
                  for i in range(len(analyzer.reconstruction_errors) - window)]
        task_succ = [np.mean(analyzer.task_outcomes[i:i+window]) 
                    for i in range(len(analyzer.task_outcomes) - window)]
        ax.plot(rec_acc, label='Reconstruction Acc')
        ax.plot(task_succ, label='Task Success')
        ax.set_title(f'Performance Over Time (Correlation: {metrics.comm_task_correlation:.3f})')
        ax.set_xlabel('Time (windows)')
        ax.set_ylabel('Rate')
        ax.legend()
    
    # Emergent patterns
    ax = axes[1, 1]
    if metrics.emergent_patterns:
        patterns = list(metrics.emergent_patterns.keys())[:5]
        counts = [metrics.emergent_patterns[p] for p in patterns]
        ax.bar(patterns, counts)
        ax.set_title('Top Emergent Patterns')
        ax.set_xlabel('Pattern')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    return fig


if __name__ == "__main__":
    # Example usage
    analyzer = CommunicationAnalyzer(vocab_size=32, max_message_length=8)
    reward_fn = CommunicationReward()
    
    # Simulate some communications
    for i in range(100):
        # Random message
        message = torch.randint(0, 32, (8,))
        reconstruction_error = np.random.random() * 0.5
        task_success = np.random.random() > 0.3
        
        analyzer.record_message(message, reconstruction_error, task_success)
    
    # Compute metrics
    metrics = analyzer.compute_metrics()
    print(f"Communication Metrics:")
    print(f"  Avg Message Length: {metrics.avg_message_length:.2f}")
    print(f"  Bandwidth Usage: {metrics.bandwidth_usage:.2f} bits")
    print(f"  Vocabulary Entropy: {metrics.vocabulary_entropy:.2f}")
    print(f"  Task Success Rate: {metrics.task_success_rate:.2%}")
    print(f"  Emergent Patterns: {metrics.emergent_patterns}")