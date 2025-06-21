# janus/environments/ai_interpretability/transformer_env.py
"""
Refactored Transformer Interpretability Environment
==================================================

Specialized environment for interpreting transformer models.
"""

import numpy as np
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass

from janus_ai.environments.ai_interpretability.neural_net_env import (
    AIInterpretabilityEnv, AIBehaviorData
)
from janus_ai.core.expressions.expression import Variable
from janus_ai.ai_interpretability.grammars.transformer_grammar import TransformerGrammar


@dataclass
class TransformerBehaviorData(AIBehaviorData):
    """Extended behavior data for transformer models."""
    token_embeddings: Optional[np.ndarray] = None  # Shape: (n_samples, seq_len, embed_dim)
    positional_encodings: Optional[np.ndarray] = None
    attention_maps: Optional[Dict[str, np.ndarray]] = None  # layer -> (n_samples, n_heads, seq_len, seq_len)
    layer_outputs: Optional[Dict[str, np.ndarray]] = None


class TransformerInterpretabilityEnv(AIInterpretabilityEnv):
    """
    Environment for discovering interpretable patterns in transformer models.
    
    Focuses on attention mechanisms and sequential relationships.
    """
    
    def __init__(
        self,
        transformer_model: torch.nn.Module,
        grammar: TransformerGrammar,
        behavior_data: TransformerBehaviorData,
        target_layer: Optional[int] = None,
        target_head: Optional[int] = None,
        sequence_position: Optional[int] = None,
        interpretation_target: str = 'attention',  # 'attention', 'output', 'embedding'
        **kwargs
    ):
        """
        Initialize transformer interpretability environment.
        
        Args:
            transformer_model: The transformer model to interpret
            grammar: Transformer-specific grammar
            behavior_data: Transformer behavior data including attention
            target_layer: Which layer to focus on (None for all)
            target_head: Which attention head to explain (None for all)
            sequence_position: Which position to explain (None for all)
            interpretation_target: What aspect to interpret
            **kwargs: Additional arguments for parent class
        """
        self.transformer_model = transformer_model
        self.target_layer = target_layer
        self.target_head = target_head
        self.sequence_position = sequence_position
        self.interpretation_target = interpretation_target
        
        # Store original transformer data
        self.transformer_data = behavior_data
        
        # Convert to standard AIBehaviorData for parent class
        standard_behavior_data = self._prepare_standard_behavior_data()
        
        super().__init__(
            ai_model=transformer_model,
            grammar=grammar,
            behavior_data=standard_behavior_data,
            interpretation_mode='modular',  # Transformers are inherently modular
            **kwargs
        )
    
    def _prepare_standard_behavior_data(self) -> AIBehaviorData:
        """Convert transformer-specific data to standard format."""
        # Determine what we're trying to explain
        if self.interpretation_target == 'attention':
            inputs, outputs = self._prepare_attention_data()
        elif self.interpretation_target == 'output':
            inputs, outputs = self._prepare_output_data()
        else:  # 'embedding'
            inputs, outputs = self._prepare_embedding_data()
        
        # Create standard behavior data
        return AIBehaviorData(
            inputs=inputs,
            outputs=outputs,
            intermediate_activations=self.transformer_data.layer_outputs,
            attention_weights=self._extract_target_attention(),
            metadata={
                'interpretation_target': self.interpretation_target,
                'target_layer': self.target_layer,
                'target_head': self.target_head,
                'sequence_position': self.sequence_position
            }
        )
    
    def _prepare_attention_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for explaining attention patterns."""
        # Use token embeddings as input
        token_embeds = self.transformer_data.token_embeddings
        n_samples, seq_len, embed_dim = token_embeds.shape
        
        # Flatten or select based on sequence position
        if self.sequence_position is not None:
            # Explain attention for specific position
            X_data = token_embeds[:, self.sequence_position, :]
        else:
            # Use mean pooling over sequence
            X_data = token_embeds.mean(axis=1)
        
        # Target is attention weights
        attention_weights = self._extract_target_attention()
        
        if attention_weights.ndim > 2:
            # Flatten attention to 2D
            y_data = attention_weights.reshape(n_samples, -1)
        else:
            y_data = attention_weights
        
        return X_data, y_data
    
    def _prepare_output_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for explaining model outputs."""
        # Use embeddings + attention as features
        token_embeds = self.transformer_data.token_embeddings
        n_samples = token_embeds.shape[0]
        
        # Create feature vector combining embeddings and attention
        features = []
        
        # Add pooled embeddings
        features.append(token_embeds.mean(axis=1))
        
        # Add attention statistics
        if self.transformer_data.attention_maps:
            for layer_name, attn_map in self.transformer_data.attention_maps.items():
                # Use attention entropy or other statistics as features
                attn_entropy = -np.sum(attn_map * np.log(attn_map + 1e-10), axis=-1)
                features.append(attn_entropy.reshape(n_samples, -1).mean(axis=1, keepdims=True))
        
        X_data = np.hstack(features)
        y_data = self.transformer_data.outputs
        
        return X_data, y_data
    
    def _prepare_embedding_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for explaining embedding patterns."""
        # Use input tokens as features (need to be provided separately)
        if hasattr(self.transformer_data, 'input_ids'):
            # One-hot encode or embed input tokens
            X_data = self._encode_tokens(self.transformer_data.input_ids)
        else:
            # Fall back to using initial embeddings
            X_data = self.transformer_data.inputs
        
        # Target is the token embeddings
        token_embeds = self.transformer_data.token_embeddings
        
        if self.sequence_position is not None:
            y_data = token_embeds[:, self.sequence_position, :]
        else:
            y_data = token_embeds.mean(axis=1)
        
        return X_data, y_data
    
    def _extract_target_attention(self) -> np.ndarray:
        """Extract attention weights for the target layer/head."""
        if not self.transformer_data.attention_maps:
            # Return dummy data if no attention maps
            return np.zeros((len(self.transformer_data.inputs), 1))
        
        # Get attention for specified layer
        if self.target_layer is not None:
            layer_key = f'layer_{self.target_layer}'
            if layer_key in self.transformer_data.attention_maps:
                attention = self.transformer_data.attention_maps[layer_key]
            else:
                # Fall back to first available layer
                attention = next(iter(self.transformer_data.attention_maps.values()))
        else:
            # Average across all layers
            all_attention = list(self.transformer_data.attention_maps.values())
            attention = np.stack(all_attention).mean(axis=0)
        
        # Select specific head if requested
        if self.target_head is not None and attention.ndim >= 3:
            attention = attention[:, self.target_head, :, :]
        
        # Select specific position if requested
        if self.sequence_position is not None and attention.ndim >= 2:
            if attention.ndim == 4:  # (batch, heads, seq, seq)
                attention = attention[:, :, self.sequence_position, :]
            elif attention.ndim == 3:  # (batch, seq, seq)
                attention = attention[:, self.sequence_position, :]
        
        return attention
    
    def _extract_variables_from_model(self) -> List[Variable]:
        """Extract transformer-specific variables."""
        variables = super()._extract_variables_from_model()
        
        # Add transformer-specific variables based on interpretation target
        if self.interpretation_target == 'attention':
            # Add position-based variables
            if hasattr(self, 'transformer_data') and self.transformer_data.positional_encodings is not None:
                pos_enc = self.transformer_data.positional_encodings
                for i in range(pos_enc.shape[-1]):
                    var = Variable(
                        name=f'pos_enc_{i}',
                        index=len(variables) + i,
                        properties={
                            'type': 'positional',
                            'dimension': i,
                            'description': f'Positional encoding dimension {i}'
                        }
                    )
                    variables.append(var)
        
        return variables
    
    def _encode_tokens(self, input_ids: np.ndarray) -> np.ndarray:
        """Encode token IDs as features."""
        # Simple one-hot encoding (could be improved)
        vocab_size = int(input_ids.max()) + 1
        n_samples = input_ids.shape[0]
        
        if input_ids.ndim == 1:
            # Single position
            encoded = np.zeros((n_samples, vocab_size))
            encoded[np.arange(n_samples), input_ids] = 1
        else:
            # Multiple positions - use bag of words
            encoded = np.zeros((n_samples, vocab_size))
            for i in range(n_samples):
                unique, counts = np.unique(input_ids[i], return_counts=True)
                encoded[i, unique] = counts
        
        return encoded