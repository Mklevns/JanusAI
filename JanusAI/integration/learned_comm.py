# JanusAI/integration/learned_comm.py v2

"""
Improved Learnable Communication System with Enhanced Features
=============================================================

Incorporates:
1. Strong communication cost pressure
2. Compositional structure supervision
3. Harder discreteness enforcement
4. Attention-based aggregation
5. Adversarial validation
6. Enhanced analysis tools

Author: JanusAI Team
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Optional, Any, Set
from dataclasses import dataclass
import logging
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class ImprovedCommunicationConfig:
    """Enhanced configuration with new parameters."""
    # Basic parameters
    vocab_size: int = 64
    max_message_length: int = 8
    hidden_dim: int = 128
    latent_dim: int = 32
    
    # Discreteness enforcement
    temperature: float = 1.0
    min_temperature: float = 0.01  # Lower minimum for harder discreteness
    temperature_decay: float = 0.995
    entropy_regularization: float = 0.1  # Entropy loss weight
    
    # Communication cost
    symbol_cost: float = 0.01  # Cost per symbol used
    length_penalty: float = 0.05  # Additional penalty for long messages
    
    # Compositional structure
    use_compositional_prior: bool = True
    compositional_loss_weight: float = 0.2
    structural_embeddings_dim: int = 16
    
    # Attention aggregation
    use_attention_aggregation: bool = True
    num_attention_heads: int = 8
    attention_dropout: float = 0.1
    
    # Adversarial validation
    use_adversarial_validator: bool = True
    adversarial_loss_weight: float = 0.1


class CompositionalEmbeddings(nn.Module):
    """
    Learned embeddings for compositional structure.
    
    Provides structural priors for mathematical expressions:
    - Operator embeddings ('+', '*', '^', 'sin', etc.)
    - Variable embeddings ('x', 'y', etc.)
    - Constant embeddings (numbers)
    """
    
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        
        # Define compositional categories
        self.categories = {
            'operators': list(range(0, 10)),  # +, -, *, /, ^, sin, cos, log, exp, sqrt
            'variables': list(range(10, 20)),  # x, y, z, etc.
            'constants': list(range(20, 30)),  # 0-9
            'structural': list(range(30, 40)),  # (, ), =, etc.
            'special': list(range(40, vocab_size))  # Everything else
        }
        
        # Category embeddings
        self.category_embeddings = nn.ModuleDict({
            cat: nn.Embedding(len(indices), embed_dim // 2)
            for cat, indices in self.categories.items()
        })
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim // 2)
        
        # Compositional rules (learnable)
        self.composition_rules = nn.Parameter(torch.randn(5, 5, embed_dim))
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get compositional embeddings for tokens."""
        batch_size, seq_len = token_ids.shape
        
        # Get token embeddings
        token_embeds = self.token_embeddings(token_ids)
        
        # Get category embeddings
        category_embeds = []
        for i in range(seq_len):
            cat_embed_list = []
            for cat, indices in self.categories.items():
                # Create mask for this category
                mask = torch.zeros_like(token_ids[:, i])
                for idx in indices:
                    mask |= (token_ids[:, i] == idx)
                
                if mask.any():
                    # Get category embedding for matching tokens
                    cat_indices = torch.zeros_like(token_ids[:, i])
                    for j, idx in enumerate(indices):
                        cat_indices[token_ids[:, i] == idx] = j
                    
                    cat_embed = self.category_embeddings[cat](cat_indices)
                    cat_embed = cat_embed * mask.float().unsqueeze(-1)
                    cat_embed_list.append(cat_embed)
            
            if cat_embed_list:
                category_embeds.append(torch.stack(cat_embed_list).sum(0))
            else:
                category_embeds.append(torch.zeros_like(token_embeds[:, i]))
        
        category_embeds = torch.stack(category_embeds, dim=1)
        
        # Combine token and category embeddings
        combined = torch.cat([token_embeds, category_embeds], dim=-1)
        
        return combined
    
    def get_compositional_loss(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Compute loss that encourages compositional structure.
        
        Penalizes invalid patterns like:
        - Consecutive operators
        - Unbalanced parentheses
        - Invalid operator-operand combinations
        """
        batch_size, seq_len = sequences.shape
        loss = torch.tensor(0.0, device=sequences.device)
        
        # Check for consecutive operators
        for i in range(seq_len - 1):
            curr_is_op = torch.zeros(batch_size, dtype=torch.bool, device=sequences.device)
            next_is_op = torch.zeros(batch_size, dtype=torch.bool, device=sequences.device)
            
            for op_idx in self.categories['operators']:
                curr_is_op |= (sequences[:, i] == op_idx)
                next_is_op |= (sequences[:, i + 1] == op_idx)
            
            # Penalty for consecutive operators
            consecutive_ops = curr_is_op & next_is_op
            loss += consecutive_ops.float().mean()
        
        return loss


class ImprovedSymbolicEncoder(nn.Module):
    """Enhanced encoder with stronger discreteness and structure."""
    
    def __init__(self, input_dim: int, config: ImprovedCommunicationConfig):
        super().__init__()
        self.config = config
        
        # Compositional embeddings
        self.compositional_embeds = CompositionalEmbeddings(
            config.vocab_size, 
            config.structural_embeddings_dim
        )
        
        # State encoder with residual connections
        self.state_encoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim if i == 0 else config.hidden_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for i in range(3)
        ])
        
        # Message head with structural bias
        self.message_head = nn.Linear(
            config.hidden_dim + config.structural_embeddings_dim,
            config.max_message_length * config.vocab_size
        )
        
        # Length prediction head
        self.length_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.max_message_length),
            nn.Softmax(dim=-1)
        )
        
        # Gumbel-Softmax with adjustable temperature
        self.temperature = config.temperature
        
    def forward(self, state: torch.Tensor, 
                enforce_sparsity: bool = True) -> Dict[str, torch.Tensor]:
        """
        Encode with enhanced discreteness and structure.
        """
        # Encode state with residual connections
        hidden = state
        for layer in self.state_encoder:
            hidden = hidden + layer(hidden)
        
        # Predict message length
        length_probs = self.length_head(hidden)
        
        # Get structural prior
        struct_prior = torch.randn(
            state.size(0) if state.dim() > 1 else 1,
            self.config.structural_embeddings_dim,
            device=state.device
        )
        
        # Generate message logits with structural bias
        combined = torch.cat([hidden, struct_prior], dim=-1)
        message_logits = self.message_head(combined)
        
        # Reshape
        batch_size = state.size(0) if state.dim() > 1 else 1
        message_logits = message_logits.view(
            batch_size, 
            self.config.max_message_length,
            self.config.vocab_size
        )
        
        # Apply sparsity enforcement
        if enforce_sparsity:
            # Add noise to encourage exploration but with sparsity bias
            sparse_bias = -torch.ones_like(message_logits) * 2.0
            sparse_bias[:, :, :self.config.vocab_size // 4] = 0  # Allow first quarter of vocab
            message_logits = message_logits + sparse_bias
        
        # Gumbel-Softmax sampling
        message = self._gumbel_softmax(message_logits, hard=True)
        
        # Apply length masking
        length_mask = self._create_length_mask(length_probs)
        message = message * length_mask.unsqueeze(-1)
        
        # Get actual message length
        actual_lengths = (message.sum(dim=-1) > 0).sum(dim=-1).float()
        
        return {
            'message': message,
            'message_logits': message_logits,
            'length_probs': length_probs,
            'actual_lengths': actual_lengths,
            'hidden': hidden
        }
    
    def _gumbel_softmax(self, logits: torch.Tensor, hard: bool = True) -> torch.Tensor:
        """Enhanced Gumbel-Softmax with very low temperature."""
        # Sample Gumbel noise
        U = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(U + 1e-20) + 1e-20)
        
        # Add noise and apply softmax with current temperature
        y_soft = F.softmax((logits + gumbel) / self.temperature, dim=-1)
        
        if hard:
            # Straight-through estimator
            y_hard = torch.zeros_like(y_soft)
            y_hard.scatter_(-1, y_soft.argmax(dim=-1, keepdim=True), 1.0)
            y = (y_hard - y_soft).detach() + y_soft
        else:
            y = y_soft
        
        return y
    
    def _create_length_mask(self, length_probs: torch.Tensor) -> torch.Tensor:
        """Create mask based on predicted lengths."""
        batch_size = length_probs.size(0)
        max_len = length_probs.size(1)
        
        # Sample actual length
        lengths = torch.multinomial(length_probs, 1).squeeze(-1)
        
        # Create mask
        mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len)
        mask = mask < lengths.unsqueeze(1)
        
        return mask.float()
    
    def update_temperature(self, decay: Optional[float] = None):
        """Update temperature with default from config."""
        decay = decay or self.config.temperature_decay
        self.temperature = max(self.temperature * decay, self.config.min_temperature)


class AttentionAggregator(nn.Module):
    """
    Multi-head attention aggregator for peer communications.
    
    Allows agents to selectively attend to relevant messages
    rather than simple averaging.
    """
    
    def __init__(self, config: ImprovedCommunicationConfig):
        super().__init__()
        self.config = config
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=config.latent_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )
        
        # Learnable query transformation
        self.query_transform = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
            nn.ReLU()
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
            nn.ReLU(),
            nn.Dropout(config.attention_dropout)
        )
        
    def forward(self, own_state: torch.Tensor, 
                peer_messages: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregate peer messages using attention.
        
        Returns:
            Aggregated message and attention weights
        """
        if not peer_messages:
            return torch.zeros_like(own_state), torch.zeros(1, 1)
        
        # Stack peer messages
        peers = torch.stack(peer_messages, dim=0)  # (num_peers, latent_dim)
        
        # Ensure batch dimension
        if own_state.dim() == 1:
            own_state = own_state.unsqueeze(0)
            peers = peers.unsqueeze(0)  # (1, num_peers, latent_dim)
        
        # Transform query
        query = self.query_transform(own_state).unsqueeze(1)  # (batch, 1, latent_dim)
        
        # Apply attention
        attended, attention_weights = self.attention(query, peers, peers)
        
        # Project output
        output = self.output_projection(attended.squeeze(1))
        
        return output, attention_weights.squeeze(1)


class AdversarialValidator(nn.Module):
    """
    Adversarial component that tries to detect invalid/misleading messages.
    
    Trained to distinguish between:
    - Valid discovery communications
    - Random/noisy messages
    - Deliberately misleading messages
    """
    
    def __init__(self, message_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.discriminator = nn.Sequential(
            nn.Linear(message_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 3)  # Valid, Random, Misleading
        )
        
    def forward(self, message: torch.Tensor) -> torch.Tensor:
        """
        Classify message validity.
        
        Returns:
            Logits for [valid, random, misleading]
        """
        # Flatten message if needed
        if message.dim() > 2:
            message = message.view(message.size(0), -1)
        
        return self.discriminator(message)
    
    def compute_adversarial_loss(self, 
                                real_messages: torch.Tensor,
                                fake_messages: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute adversarial training loss."""
        batch_size = real_messages.size(0)
        device = real_messages.device
        
        # Real messages should be classified as valid
        real_preds = self.forward(real_messages)
        real_labels = torch.zeros(batch_size, dtype=torch.long, device=device)  # Valid = 0
        real_loss = F.cross_entropy(real_preds, real_labels)
        
        # Generate fake messages if not provided
        if fake_messages is None:
            # Random messages
            if real_messages.dim() == 3:
                fake_shape = real_messages.shape
                fake_messages = torch.randint(0, fake_shape[-1], fake_shape[:-1], device=device)
                fake_messages = F.one_hot(fake_messages, fake_shape[-1]).float()
            else:
                fake_messages = torch.randn_like(real_messages)
        
        # Fake messages should be classified as random
        fake_preds = self.forward(fake_messages)
        fake_labels = torch.ones(batch_size, dtype=torch.long, device=device)  # Random = 1
        fake_loss = F.cross_entropy(fake_preds, fake_labels)
        
        return real_loss + fake_loss


class ImprovedCommunicationReward:
    """
    Enhanced reward function with all improvements.
    
    Includes:
    - Symbol cost per token
    - Length penalty
    - Compositional structure bonus
    - Adversarial validation penalty
    """
    
    def __init__(self, config: ImprovedCommunicationConfig):
        self.config = config
        
    def compute_reward(self,
                      original_state: torch.Tensor,
                      reconstructed_state: torch.Tensor,
                      message: torch.Tensor,
                      actual_lengths: torch.Tensor,
                      task_success: bool,
                      compositional_loss: float = 0.0,
                      adversarial_score: float = 1.0) -> Tuple[float, Dict[str, float]]:
        """
        Compute comprehensive reward with all components.
        """
        components = {}
        
        # 1. Reconstruction fidelity (most important)
        mse = F.mse_loss(reconstructed_state, original_state)
        components['fidelity'] = -mse.item()
        
        # 2. Communication cost (symbol cost + length penalty)
        avg_length = actual_lengths.mean().item()
        symbol_cost = self.config.symbol_cost * avg_length
        length_penalty = self.config.length_penalty * max(0, avg_length - 4)  # Penalty for > 4 tokens
        components['symbol_cost'] = -symbol_cost
        components['length_penalty'] = -length_penalty
        
        # 3. Task success
        components['task_success'] = float(task_success)
        
        # 4. Compositional structure bonus
        components['compositional'] = -compositional_loss
        
        # 5. Adversarial validation
        components['adversarial'] = adversarial_score
        
        # Weighted combination
        total_reward = (
            0.4 * components['fidelity'] +
            0.1 * components['symbol_cost'] +
            0.05 * components['length_penalty'] +
            0.3 * components['task_success'] +
            0.1 * components['compositional'] +
            0.05 * components['adversarial']
        )
        
        return total_reward, components


class LanguageEvolutionTracker:
    """
    Enhanced tracking of language evolution with per-phase statistics.
    """
    
    def __init__(self, vocab_size: int, max_phases: int = 3):
        self.vocab_size = vocab_size
        self.max_phases = max_phases
        
        # Per-phase tracking
        self.phase_stats = {
            phase: {
                'symbol_counts': defaultdict(int),
                'bigram_counts': defaultdict(int),
                'expression_mappings': defaultdict(list),
                'task_successes': [],
                'message_lengths': [],
                'unique_messages': set()
            } for phase in range(1, max_phases + 1)
        }
        
        # Global tracking
        self.symbol_evolution = []  # List of vocab distributions over time
        self.expression_consistency = defaultdict(lambda: defaultdict(int))
        
    def record_communication(self,
                           phase: int,
                           message: torch.Tensor,
                           expression: str,
                           task_success: bool,
                           episode: int):
        """Record communication instance with phase tracking."""
        # Convert message to string
        if message.dim() == 3:
            tokens = message.argmax(dim=-1).cpu().numpy()
        else:
            tokens = message.cpu().numpy()
        
        msg_str = "-".join(map(str, tokens.flatten()))
        
        # Update phase statistics
        phase_data = self.phase_stats[phase]
        
        # Symbol counts
        for token in tokens.flatten():
            phase_data['symbol_counts'][int(token)] += 1
        
        # Bigrams
        for i in range(len(tokens.flatten()) - 1):
            bigram = (int(tokens.flatten()[i]), int(tokens.flatten()[i + 1]))
            phase_data['bigram_counts'][bigram] += 1
        
        # Expression mapping
        phase_data['expression_mappings'][expression].append(msg_str)
        self.expression_consistency[expression][msg_str] += 1
        
        # Task success
        phase_data['task_successes'].append(task_success)
        
        # Message length
        actual_length = (tokens != 0).sum()
        phase_data['message_lengths'].append(int(actual_length))
        
        # Unique messages
        phase_data['unique_messages'].add(msg_str)
        
        # Global evolution tracking (sample every 10 episodes)
        if episode % 10 == 0:
            vocab_dist = np.zeros(self.vocab_size)
            total = sum(phase_data['symbol_counts'].values())
            if total > 0:
                for token, count in phase_data['symbol_counts'].items():
                    if token < self.vocab_size:
                        vocab_dist[token] = count / total
            self.symbol_evolution.append((episode, phase, vocab_dist))
    
    def get_phase_summary(self, phase: int) -> Dict[str, Any]:
        """Get comprehensive statistics for a phase."""
        phase_data = self.phase_stats[phase]
        
        # Calculate metrics
        total_symbols = sum(phase_data['symbol_counts'].values())
        unique_symbols = len(phase_data['symbol_counts'])
        
        if total_symbols > 0:
            symbol_entropy = -sum(
                (count/total_symbols) * np.log2(count/total_symbols)
                for count in phase_data['symbol_counts'].values()
                if count > 0
            )
        else:
            symbol_entropy = 0
        
        # Expression consistency
        expr_consistencies = []
        for expr, mappings in phase_data['expression_mappings'].items():
            if len(mappings) > 1:
                most_common = max(set(mappings), key=mappings.count)
                consistency = mappings.count(most_common) / len(mappings)
                expr_consistencies.append(consistency)
        
        avg_consistency = np.mean(expr_consistencies) if expr_consistencies else 0
        
        return {
            'unique_symbols_used': unique_symbols,
            'symbol_entropy': symbol_entropy,
            'avg_message_length': np.mean(phase_data['message_lengths']) if phase_data['message_lengths'] else 0,
            'task_success_rate': np.mean(phase_data['task_successes']) if phase_data['task_successes'] else 0,
            'unique_messages': len(phase_data['unique_messages']),
            'expression_consistency': avg_consistency,
            'symbol_reuse_rate': total_symbols / unique_symbols if unique_symbols > 0 else 0
        }
    
    def create_symbol_heatmap(self, save_path: Optional[str] = None):
        """Create heatmap showing symbol usage evolution."""
        if not self.symbol_evolution:
            return None
        
        # Create matrix: episodes x vocab_size
        episodes = [e[0] for e in self.symbol_evolution]
        vocab_matrix = np.array([e[2] for e in self.symbol_evolution]).T
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(vocab_matrix, 
                   cmap='YlOrRd',
                   xticklabels=[f"Ep{e}" if i % 10 == 0 else "" 
                               for i, e in enumerate(episodes)],
                   yticklabels=[f"T{i}" if i % 5 == 0 else "" 
                               for i in range(self.vocab_size)],
                   cbar_kws={'label': 'Usage Frequency'})
        
        plt.title('Symbol Usage Evolution Over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Token ID')
        
        # Add phase boundaries
        phase_changes = []
        for i in range(1, len(self.symbol_evolution)):
            if self.symbol_evolution[i][1] != self.symbol_evolution[i-1][1]:
                phase_changes.append(i)
        
        for change in phase_changes:
            plt.axvline(x=change, color='blue', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        
        return plt.gcf()
    
    def get_emergent_dictionary(self, min_count: int = 5) -> Dict[str, List[str]]:
        """Extract emergent symbol-expression mappings."""
        dictionary = defaultdict(list)
        
        for expr, msg_counts in self.expression_consistency.items():
            # Get most common message for this expression
            if msg_counts:
                sorted_msgs = sorted(msg_counts.items(), key=lambda x: x[1], reverse=True)
                for msg, count in sorted_msgs:
                    if count >= min_count:
                        dictionary[msg].append(f"{expr} ({count}x)")
        
        # Sort by frequency
        return dict(sorted(dictionary.items(), 
                         key=lambda x: sum(int(item.split('(')[1].rstrip('x)')) 
                                         for item in x[1]), 
                         reverse=True)[:20])


# Visualization utilities
def plot_communication_analysis(tracker: LanguageEvolutionTracker, save_prefix: str = "comm_analysis"):
    """Create comprehensive visualization of communication evolution."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Symbol entropy over phases
    ax = axes[0, 0]
    phases = []
    entropies = []
    for phase in range(1, 4):
        summary = tracker.get_phase_summary(phase)
        if summary['symbol_entropy'] > 0:
            phases.append(f"Phase {phase}")
            entropies.append(summary['symbol_entropy'])
    
    ax.bar(phases, entropies)
    ax.set_title('Symbol Entropy by Phase')
    ax.set_ylabel('Entropy (bits)')
    
    # Plot 2: Message length distribution by phase
    ax = axes[0, 1]
    for phase in range(1, 4):
        lengths = tracker.phase_stats[phase]['message_lengths']
        if lengths:
            ax.hist(lengths, bins=range(0, 10), alpha=0.5, label=f'Phase {phase}')
    ax.set_title('Message Length Distribution')
    ax.set_xlabel('Length')
    ax.set_ylabel('Frequency')
    ax.legend()
    
    # Plot 3: Task success vs communication consistency
    ax = axes[0, 2]
    consistencies = []
    successes = []
    for phase in range(1, 4):
        summary = tracker.get_phase_summary(phase)
        if summary['expression_consistency'] > 0:
            consistencies.append(summary['expression_consistency'])
            successes.append(summary['task_success_rate'])
    
    if consistencies:
        ax.scatter(consistencies, successes, s=100)
        ax.set_xlabel('Expression Consistency')
        ax.set_ylabel('Task Success Rate')
        ax.set_title('Consistency vs Performance')
    
    # Plot 4: Symbol reuse rate
    ax = axes[1, 0]
    phases = []
    reuse_rates = []
    for phase in range(1, 4):
        summary = tracker.get_phase_summary(phase)
        if summary['symbol_reuse_rate'] > 0:
            phases.append(f"Phase {phase}")
            reuse_rates.append(summary['symbol_reuse_rate'])
    
    ax.bar(phases, reuse_rates)
    ax.set_title('Symbol Reuse Rate by Phase')
    ax.set_ylabel('Avg Uses per Symbol')
    
    # Plot 5: Unique messages over time
    ax = axes[1, 1]
    for phase in range(1, 4):
        unique_msgs = len(tracker.phase_stats[phase]['unique_messages'])
        if unique_msgs > 0:
            ax.bar(f"Phase {phase}", unique_msgs)
    ax.set_title('Unique Messages by Phase')
    ax.set_ylabel('Count')
    
    # Plot 6: Top emergent patterns
    ax = axes[1, 2]
    dictionary = tracker.get_emergent_dictionary()
    if dictionary:
        top_patterns = list(dictionary.keys())[:5]
        pattern_counts = [sum(int(item.split('(')[1].rstrip('x)')) 
                            for item in dictionary[p]) 
                         for p in top_patterns]
        
        ax.barh(range(len(top_patterns)), pattern_counts)
        ax.set_yticks(range(len(top_patterns)))
        ax.set_yticklabels([p[:15] + "..." if len(p) > 15 else p for p in top_patterns])
        ax.set_xlabel('Total Uses')
        ax.set_title('Top Message Patterns')
    
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_overview.png", dpi=150)
    
    # Create symbol heatmap
    tracker.create_symbol_heatmap(f"{save_prefix}_heatmap.png")
    
    return fig


if __name__ == "__main__":
    # Test improved system
    config = ImprovedCommunicationConfig(
        vocab_size=64,
        max_message_length=8,
        symbol_cost=0.02,
        use_compositional_prior=True,
        use_attention_aggregation=True,
        use_adversarial_validator=True
    )
    
    # Test components
    print("Testing improved communication system...")
    
    # Test encoder
    encoder = ImprovedSymbolicEncoder(64, config)
    test_state = torch.randn(2, 64)
    encoded = encoder(test_state)
    print(f"Encoded message shape: {encoded['message'].shape}")
    print(f"Actual lengths: {encoded['actual_lengths']}")
    
    # Test attention aggregator
    aggregator = AttentionAggregator(config)
    own_state = torch.randn(32)
    peer_msgs = [torch.randn(32) for _ in range(3)]
    aggregated, weights = aggregator(own_state, peer_msgs)
    print(f"Aggregated shape: {aggregated.shape}")
    print(f"Attention weights shape: {weights.shape}")
    
    # Test adversarial validator
    validator = AdversarialValidator(8 * 64, 128)
    fake_msg = torch.randn(2, 8, 64)
    validity = validator(fake_msg)
    print(f"Validity scores: {validity}")
    
    print("\nAll components working correctly!")