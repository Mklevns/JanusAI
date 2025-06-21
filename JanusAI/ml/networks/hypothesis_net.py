    # Test AIHypothesisNet with Transformer encoder and Meta-learning enabled
"""
HypothesisNet Policy Network
============================

Neural policy for intelligent hypothesis generation in the SymbolicDiscoveryEnv.
Implements both TreeLSTM and Transformer variants with action masking.
This file contains the core neural network architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, TypedDict
from dataclasses import dataclass

# Internal project imports based on new structure
from janus_ai.environments.base.symbolic_env import SymbolicDiscoveryEnv, safe_env_reset
from janus_ai.core.grammar.progressive_grammar import ProgressiveGrammar # Updated import
from janus_ai.core.expressions.expression import Variable
from janus_ai.utils.general_utils import safe_import # Assuming this new utility file will be created


# Use safe_import for wandb. It's conditionally imported here but not directly used within the classes themselves.
# The original file had `wandb = safe_import("wandb", "wandb")`
# For a clean separation, if wandb logging is only handled by the trainer,
# it might not strictly need to be imported here.
# However, if any policy-level logging or visualization directly depends on wandb, it should remain.
# For now, keeping it as it was in the original file, assuming it might be used by the policy for internal logging if enabled.
wandb = safe_import("wandb", "wandb")


class HypothesisNetOutput(TypedDict, total=False):
    """Output dictionary from HypothesisNet forward pass."""
    action_logits: torch.Tensor
    value: torch.Tensor
    tree_representation: torch.Tensor
    task_embedding: torch.Tensor


class TreeLSTMCell(nn.Module):
    """Tree-structured LSTM cell for processing expression trees."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input gate
        self.W_i = nn.Linear(input_dim, hidden_dim)
        self.U_i = nn.Linear(hidden_dim, hidden_dim)
        # Forget gate (for each child) - Assuming two children for simplicity,
        # adjust for N-ary trees if needed
        self.W_f = nn.Linear(input_dim, hidden_dim)
        self.U_f = nn.Linear(hidden_dim, hidden_dim)
        # Output gate
        self.W_o = nn.Linear(input_dim, hidden_dim)
        self.U_o = nn.Linear(hidden_dim, hidden_dim)
        # Cell gate
        self.W_c = nn.Linear(input_dim, hidden_dim)
        self.U_c = nn.Linear(hidden_dim, hidden_dim)

    def forward(self,
                x: torch.Tensor,
                children_h: List[torch.Tensor],
                children_c: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for TreeLSTMCell.

        Args:
            x: Input features for the current node.
            children_h: List of hidden states of children nodes.
            children_c: List of cell states of children nodes.

        Returns:
            Tuple of (new hidden state, new cell state).
        """
        # Sum of children's hidden states for input, output, and cell gates
        if not children_h:
            h_sum = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        else:
            h_sum = sum(children_h)

        i = torch.sigmoid(self.W_i(x) + self.U_i(h_sum))
        o = torch.sigmoid(self.W_o(x) + self.U_o(h_sum))
        c_tilde = torch.tanh(self.W_c(x) + self.U_c(h_sum))

        # Initialize new cell state with input gate and candidate cell state
        c = i * c_tilde
        # Apply forget gates for each child and add to cell state
        if children_c:
            for child_c, child_h in zip(children_c, children_h):
                f = torch.sigmoid(self.W_f(x) + self.U_f(child_h))
                c = c + f * child_c
        
        h = o * torch.tanh(c)
        return h, c


class TreeEncoder(nn.Module):
    """
    Encodes expression trees using a Tree-structured LSTM.
    Processes nodes recursively based on the tree structure.
    """
    def __init__(self, node_feature_dim: int, hidden_dim: int, n_layers: int = 2):
        super().__init__()
        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)
        # A list of TreeLSTMCell, one for each layer
        self.tree_cells = nn.ModuleList([
            TreeLSTMCell(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    def forward(self, tree_features: torch.Tensor, tree_structure: Optional[Dict[int, List[int]]] = None) -> torch.Tensor:
        """
        Forward pass for TreeEncoder.

        Args:
            tree_features: Tensor of node features (batch_size, max_nodes, node_feature_dim).
            tree_structure: Optional dictionary mapping parent node index to a list of child node indices.
                            If None, assumes a simple sequential processing (not ideal for actual trees).

        Returns:
            Latent representation of the tree (batch_size, hidden_dim).
        """
        batch_size, max_nodes, _ = tree_features.shape
        node_embeds = self.node_embedding(tree_features) # (batch_size, max_nodes, hidden_dim)

        if tree_structure is None:
            # Fallback to sequential processing if no tree structure is provided
            # This is a simplification and might not capture true tree dependencies
            h_prev_layer = node_embeds
            for cell in self.tree_cells:
                h_current_layer_nodes = []
                for i in range(max_nodes):
                    # For sequential, pass empty children list and assume input is direct
                    h_node, _ = cell(h_prev_layer[:, i, :], [], [])
                    h_current_layer_nodes.append(h_node)
                h_prev_layer = torch.stack(h_current_layer_nodes, dim=1)
            # Return mean pooling of all node representations
            return torch.mean(h_prev_layer, dim=1)

        # Structured processing (recursive, implies post-order traversal)
        # Memoization dictionary to store (h, c) for each node at each layer
        # Key: (layer_idx, node_idx), Value: (h_tensor, c_tensor)
        memo: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}

        def get_hc(layer_idx: int, node_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            """Recursive helper to compute (h, c) for a node at a given layer."""
            # If already computed, return from memo
            if (layer_idx, node_idx) in memo:
                return memo[(layer_idx, node_idx)]

            # Get the input feature for the current cell
            if layer_idx == 0:
                # For the first layer, input is the node's initial embedding
                current_cell_input = node_embeds[:, node_idx, :]
            else:
                # For subsequent layers, input is the hidden state (h) from the *previous layer* for the same node
                current_cell_input, _ = get_hc(layer_idx - 1, node_idx) # Get h from layer-1

            # Recursively get hidden and cell states of children for the *current layer*
            children_indices = tree_structure.get(node_idx, [])
            current_layer_children_h: List[torch.Tensor] = []
            current_layer_children_c: List[torch.Tensor] = []

            for child_idx in children_indices:
                h_child, c_child = get_hc(layer_idx, child_idx) # Recursive call for children
                current_layer_children_h.append(h_child)
                current_layer_children_c.append(c_child)

            # Perform the TreeLSTM cell computation
            h, c = self.tree_cells[layer_idx](current_cell_input, current_layer_children_h, current_layer_children_c)
            
            # Store result in memo
            memo[(layer_idx, node_idx)] = (h, c)
            return h, c

        # Assuming node 0 is the root or a representative node for the entire tree
        # In a batch setting, tree_structure typically applies to a single tree,
        # or it's a list of structures for a batch. Here it's a single structure.
        # To process a batch, this forward method would need to iterate through batch items.
        # Current implementation assumes a single tree structure per call.
        # If `tree_features` is batched, `tree_structure` implies a single structure applied to all.
        # A more robust solution might require `tree_structure` to be a list of structures.
        
        # To get a single representation for the whole tree, we process the root node
        # at the final layer. Assuming root is node 0.
        root_h, _ = get_hc(self.n_layers - 1, 0) # Get h from the last layer for the root node
        return root_h


class TransformerEncoder(nn.Module):
    """Encodes sequential node features using a Transformer encoder."""
    def __init__(self, node_feature_dim: int, hidden_dim: int, n_heads: int = 8, n_layers: int = 6):
        super().__init__()
        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)
        # Positional encoding for sequence order
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.hidden_dim = hidden_dim

    def forward(self, tree_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for TransformerEncoder.

        Args:
            tree_features: Tensor of node features (batch_size, max_nodes, node_feature_dim).

        Returns:
            Latent representation of the tree (batch_size, hidden_dim) by masked mean pooling.
        """
        batch_size, max_nodes, _ = tree_features.shape
        node_embeds = self.node_embedding(tree_features) # (B, N, H)
        
        # Add positional encoding
        node_embeds = node_embeds + self.positional_encoding[:, :max_nodes, :]
        
        # Create padding mask (True for padding, False for actual tokens)
        padding_mask = (tree_features.sum(dim=-1) == 0) # (B, N)
        
        # Pass through Transformer encoder
        encoded = self.transformer(node_embeds, src_key_padding_mask=padding_mask) # (B, N, H)

        # Masked mean pooling: sum only non-padded embeddings
        mask_expanded = (~padding_mask).unsqueeze(-1).float() # (B, N, 1), 1 for non-pad, 0 for pad
        sum_embeddings = (encoded * mask_expanded).sum(dim=1) # (B, H)
        # Ensure division by at least a small number to avoid NaN for empty sequences
        num_non_padded_tokens = mask_expanded.sum(dim=1).clamp(min=1e-9) # (B, 1)

        tree_repr = sum_embeddings / num_non_padded_tokens
        return tree_repr


class HypothesisNet(nn.Module):
    """
    A neural network that generates hypotheses (expressions) in an RL setting.
    It takes an observation (tree representation) and outputs action logits
    and a value estimate. Supports meta-learning through task conditioning.
    """
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        encoder_type: str = 'transformer',
        grammar: Optional[ProgressiveGrammar] = None,
        debug: bool = False,
        use_meta_learning: bool = False
    ) -> None:
        super().__init__()
        self.observation_dim = obs_dim
        self.action_dim = act_dim
        self.hidden_dim = hidden_dim
        self.grammar = grammar
        self.debug = debug
        self.use_meta_learning = use_meta_learning
        # Feature dimension for each node in the tree representation
        self.node_feature_dim = 128

        # Initialize the encoder for the tree observation
        if encoder_type == 'transformer':
            self.encoder = TransformerEncoder(self.node_feature_dim, hidden_dim)
        elif encoder_type == 'treelstm': # Added 'treelstm' option
            self.encoder = TreeEncoder(self.node_feature_dim, hidden_dim, n_layers=2)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}. Choose 'transformer' or 'treelstm'.")

        # Meta-learning components (task encoder and modulator)
        self.task_encoder = None
        self.task_modulator = None
        if self.use_meta_learning:
            self.task_encoder = nn.LSTM(
                input_size=self.node_feature_dim, hidden_size=self.hidden_dim // 2, # Note: input_size can vary based on task_trajectories format
                batch_first=True, bidirectional=True, num_layers=1
            )
            # Modulator for gains and biases for policy and value networks
            self.task_modulator = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim * 2) # Two halves for policy and value modulation
            )

        # Policy network (maps tree representation to action logits)
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        # Value network (maps tree representation to value estimate)
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Action embeddings (if needed for context, e.g., in a sequential action model)
        self.action_embeddings = nn.Embedding(action_dim, 64) # Currently unused in forward pass logic.

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Orthogonal initialization for linear layers."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        task_trajectories: Optional[torch.Tensor] = None,
        tree_structure: Optional[Dict[int, List[int]]] = None
    ) -> HypothesisNetOutput:
        """
        Forward pass through the network.

        Args:
            obs: The observation from the environment, representing the current expression tree state.
                 Expected shape: (batch_size, obs_dim_flat), where obs_dim_flat = max_nodes * node_feature_dim.
            action_mask: A boolean tensor indicating valid actions (batch_size, action_dim).
                         True means valid, False means invalid.
            task_trajectories: Optional tensor of past observations from the current task,
                               used for meta-learning to create a task embedding.
                               Shape: (batch_size, num_trajectories, trajectory_length, obs_feature_dim).
            tree_structure: Optional dictionary representing the tree structure,
                            required by TreeEncoder.

        Returns:
            Dictionary containing:
            - action_logits: Logits for action distribution after masking (batch_size, action_dim).
            - value: Value function estimate (batch_size, 1).
            - tree_representation: Latent representation of the expression tree (batch_size, hidden_dim).
            - task_embedding: Task-specific embedding (batch_size, hidden_dim), only if use_meta_learning is True.
        """
        batch_size, obs_dim_flat = obs.shape
        
        # Reshape flat observation into node features for the encoder
        if obs_dim_flat % self.node_feature_dim != 0:
            raise ValueError(f"Observation dimension ({obs_dim_flat}) must be a multiple of node_feature_dim ({self.node_feature_dim}).")
        num_nodes = obs_dim_flat // self.node_feature_dim
        tree_features = obs.view(batch_size, num_nodes, self.node_feature_dim)

        # Encode the tree features
        if isinstance(self.encoder, TreeEncoder):
            tree_repr = self.encoder(tree_features, tree_structure)
        else: # TransformerEncoder
            tree_repr = self.encoder(tree_features) # (batch_size, hidden_dim)

        task_embedding_vector = None
        # Default gains and biases if meta-learning is not active or no trajectories provided
        gains_policy = torch.ones(batch_size, self.hidden_dim // 2, device=tree_repr.device)
        biases_policy = torch.zeros(batch_size, self.hidden_dim // 2, device=tree_repr.device)
        gains_value = torch.ones(batch_size, self.hidden_dim // 2, device=tree_repr.device)
        biases_value = torch.zeros(batch_size, self.hidden_dim // 2, device=tree_repr.device)

        # Apply meta-learning (task-conditioning) if enabled
        if self.use_meta_learning and self.task_encoder and self.task_modulator and task_trajectories is not None:
            # task_trajectories are expected to be (batch_size, num_trajectories, trajectory_length, feat_dim)
            bt_size, num_traj, traj_len, feat_dim = task_trajectories.shape
            
            # Reshape for LSTM input: (batch_size * num_traj, trajectory_length, feat_dim)
            task_trajectories_reshaped = task_trajectories.view(bt_size * num_traj, traj_len, feat_dim)
            
            if task_trajectories_reshaped.shape[0] > 0: # Avoid processing empty tensors
                # Encode task trajectories using LSTM
                _, (hidden, _) = self.task_encoder(task_trajectories_reshaped)
                
                # hidden for bidirectional LSTM: (num_layers*2, batch_size*num_traj, hidden_size_of_lstm)
                # Concatenate forward and backward hidden states
                hidden_concat = torch.cat((hidden[0,:,:], hidden[1,:,:]), dim=-1) # (batch_size*num_traj, hidden_dim)
                
                # Average task embeddings over multiple trajectories per batch item
                task_embedding_vector = hidden_concat.view(bt_size, num_traj, self.hidden_dim).mean(dim=1) # (batch_size, hidden_dim)

                # Get modulation parameters from task modulator
                modulation_params = self.task_modulator(task_embedding_vector) # (batch_size, hidden_dim * 2)
                
                # Split modulation params into gains and biases for policy and value networks
                all_gains_raw, all_biases_raw = modulation_params.chunk(2, dim=-1) # Each (batch_size, hidden_dim)
                gains_policy_raw, gains_value_raw = all_gains_raw.chunk(2, dim=-1) # Each (batch_size, hidden_dim // 2)
                biases_policy_raw, biases_value_raw = all_biases_raw.chunk(2, dim=-1)

                # Apply activation and scaling for stable modulation
                gains_policy = 1 + 0.1 * torch.tanh(gains_policy_raw)
                biases_policy = 0.1 * torch.tanh(biases_policy_raw)
                gains_value = 1 + 0.1 * torch.tanh(gains_value_raw)
                biases_value = 0.1 * torch.tanh(biases_value_raw)

        # Apply policy network with optional modulation
        x_policy = tree_repr
        modulated_policy = False
        for i, layer in enumerate(self.policy_net):
            x_policy = layer(x_policy)
            if isinstance(layer, nn.ReLU) and self.use_meta_learning and task_trajectories is not None and not modulated_policy:
                # Apply modulation to the output of a ReLU layer if its dimension matches
                if x_policy.shape[-1] == self.hidden_dim // 2: # Assumes modulation is on the last ReLU output
                    x_policy = x_policy * gains_policy + biases_policy
                    modulated_policy = True # Ensure modulation is applied only once per network path
        policy_logits = x_policy

        # Apply value network with optional modulation
        x_value = tree_repr
        modulated_value = False
        for i, layer in enumerate(self.value_net):
            x_value = layer(x_value)
            if isinstance(layer, nn.ReLU) and self.use_meta_learning and task_trajectories is not None and not modulated_value:
                if x_value.shape[-1] == self.hidden_dim // 2:
                    x_value = x_value * gains_value + biases_value
                    modulated_value = True
        value = x_value

        # Apply action mask to logits
        masked_logits = policy_logits
        if action_mask is not None:
            if action_mask.shape[0] == policy_logits.shape[0]:
                masked_logits = policy_logits.clone()
                # Set logits of invalid actions to a very small number (negative infinity)
                masked_logits[~action_mask] = -1e9
            elif self.debug:
                print("Action mask shape mismatch. Skipping masking.") # Optional: for debugging

        # Handle potential NaNs or Infs from -1e9 for numerical stability
        action_logits = masked_logits.nan_to_num(nan=-1e9, posinf=1e9, neginf=-1e9)

        # Prepare output dictionary
        return_dict: HypothesisNetOutput = {
            'action_logits': action_logits,
            'value': value,
            'tree_representation': tree_repr # This is the base tree representation
        }
        if self.use_meta_learning and task_embedding_vector is not None:
            return_dict['task_embedding'] = task_embedding_vector

        return return_dict

    def get_action(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        action_mask: Optional[torch.Tensor] = None,
        task_trajectories: Optional[torch.Tensor] = None,
        tree_structure: Optional[Dict[int, List[int]]] = None,
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Selects an action based on the current observation.

        Args:
            obs: Current observation from the environment. Can be numpy array or torch tensor.
            action_mask: Boolean tensor indicating valid actions.
            task_trajectories: Optional task-specific trajectories for meta-learning.
            tree_structure: Optional tree structure for TreeEncoder.
            deterministic: If True, selects the action with the highest logit. Otherwise, samples.

        Returns:
            Tuple of (action_index, log_probability_of_action, estimated_value).
        """
        # Convert observation to a batched tensor if it's a numpy array or unbatched tensor
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        else:
            obs_tensor = obs
        
        if obs_tensor.ndim == 1: # Ensure it's (batch_size, obs_dim)
            obs_tensor = obs_tensor.unsqueeze(0)

        # Perform forward pass
        outputs = self.forward(obs_tensor, action_mask, task_trajectories, tree_structure)

        # Create categorical distribution from action logits
        dist = Categorical(logits=outputs['action_logits'])

        # Select action
        if deterministic:
            action_tensor = torch.argmax(outputs['action_logits'], dim=-1)
        else:
            action_tensor = dist.sample()
        
        # Calculate log probability and get value estimate
        log_prob_tensor = dist.log_prob(action_tensor)
        value_tensor = outputs['value']

        # Extract scalar values from tensors (assuming batch size of 1 for this method's return type)
        action_val: int = action_tensor.item()
        log_prob_val: float = log_prob_tensor.item()
        value_val: float = value_tensor.squeeze().item()

        return action_val, log_prob_val, value_val


class AIHypothesisNet(HypothesisNet):
    """
    Extends HypothesisNet to incorporate AI model-specific information for interpretation tasks.
    It can take an AI model's internal representation and its type as input.
    """
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 hidden_dim: int = 256,
                 num_model_types: int = 10, # Max number of different AI model types for embedding
                 **kwargs: Any): # Catches other args for HypothesisNet (encoder_type, grammar, debug, use_meta_learning)
        super().__init__(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim, **kwargs)

        # Multi-head attention mechanism to process AI model's internal representation
        self.model_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            batch_first=True
        )

        # Embedding layer for different AI model types
        self.model_type_embedding = nn.Embedding(
            num_embeddings=num_model_types,
            embedding_dim=self.hidden_dim
        )

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        task_trajectories: Optional[torch.Tensor] = None,
        tree_structure: Optional[Dict[int, List[int]]] = None,
        ai_model_representation: Optional[torch.Tensor] = None, # New: Input for AI model's state/features
        ai_model_type_idx: Optional[torch.Tensor] = None # New: Input for AI model type (long tensor of indices)
    ) -> HypothesisNetOutput:
        """
        Forward pass for AIHypothesisNet.
        Extends base class forward pass to incorporate AI model-specific information.

        Args:
            obs: Observation tensor from the environment.
            action_mask: Optional mask for valid actions.
            task_trajectories: Optional trajectories for meta-learning.
            tree_structure: Optional tree structure for TreeEncoder.
            ai_model_representation: A tensor representing features or state of the AI model being interpreted.
                                     Expected shape: (batch_size, sequence_length_of_model_repr, feature_dim).
                                     `feature_dim` should ideally match `self.hidden_dim`.
            ai_model_type_idx: An integer tensor (batch_size,) indicating the type of the AI model.

        Returns:
            HypothesisNetOutput dictionary.
        """
        batch_size, obs_dim_flat = obs.shape
        if obs_dim_flat % self.node_feature_dim != 0:
            raise ValueError("Observation dimension must be a multiple of node_feature_dim.")
        num_nodes = obs_dim_flat // self.node_feature_dim
        tree_features = obs.view(batch_size, num_nodes, self.node_feature_dim)

        # Encode the expression tree to get a base representation
        if isinstance(self.encoder, TreeEncoder):
            base_tree_repr = self.encoder(tree_features, tree_structure)
        else: # TransformerEncoder
            base_tree_repr = self.encoder(tree_features)

        # Initialize combined_repr with base_tree_repr
        combined_repr = base_tree_repr

        # 1. Incorporate AI model type embedding if provided
        if ai_model_type_idx is not None:
            # Ensure ai_model_type_idx is correctly shaped for embedding (batch_size)
            model_type_embed = self.model_type_embedding(ai_model_type_idx) # (batch_size, hidden_dim)
            # Combine by addition. Other methods (e.g., concatenation followed by linear layer) are possible.
            combined_repr = combined_repr + model_type_embed

        # 2. Incorporate AI model representation using attention if provided
        if ai_model_representation is not None:
            # Query the AI model representation using the current combined representation
            query = combined_repr.unsqueeze(1) # (batch_size, 1, hidden_dim)
            # Apply attention. Assuming ai_model_representation has feature_dim == hidden_dim
            attn_output, _ = self.model_attention(query, ai_model_representation, ai_model_representation)
            attn_output = attn_output.squeeze(1) # (batch_size, hidden_dim)

            # Combine attention output with the current representation
            combined_repr = combined_repr + attn_output

        # --- Remaining logic largely similar to HypothesisNet's forward pass,
        #     but operating on `combined_repr` instead of `tree_repr` directly. ---

        task_embedding_vector = None
        gains_policy = torch.ones(batch_size, self.hidden_dim // 2, device=combined_repr.device)
        biases_policy = torch.zeros(batch_size, self.hidden_dim // 2, device=combined_repr.device)
        gains_value = torch.ones(batch_size, self.hidden_dim // 2, device=combined_repr.device)
        biases_value = torch.zeros(batch_size, self.hidden_dim // 2, device=combined_repr.device)

        if self.use_meta_learning and self.task_encoder and self.task_modulator and task_trajectories is not None:
            bt_size, num_traj, traj_len, feat_dim = task_trajectories.shape
            task_trajectories_reshaped = task_trajectories.view(bt_size * num_traj, traj_len, feat_dim)
            
            if task_trajectories_reshaped.shape[0] > 0:
                _, (hidden, _) = self.task_encoder(task_trajectories_reshaped)
                hidden_concat = torch.cat((hidden[0,:,:], hidden[1,:,:]), dim=-1)
                task_embedding_vector = hidden_concat.view(bt_size, num_traj, self.hidden_dim).mean(dim=1)
                modulation_params = self.task_modulator(task_embedding_vector)
                all_gains_raw, all_biases_raw = modulation_params.chunk(2, dim=-1)
                gains_policy_raw, gains_value_raw = all_gains_raw.chunk(2, dim=-1)
                biases_policy_raw, biases_value_raw = all_biases_raw.chunk(2, dim=-1)
                gains_policy = 1 + 0.1 * torch.tanh(gains_policy_raw)
                biases_policy = 0.1 * torch.tanh(biases_policy_raw)
                gains_value = 1 + 0.1 * torch.tanh(gains_value_raw)
                biases_value = 0.1 * torch.tanh(biases_value_raw)

        # Policy network with optional modulation
        x_policy = combined_repr
        modulated_policy = False
        for i, layer in enumerate(self.policy_net):
            x_policy = layer(x_policy)
            if isinstance(layer, nn.ReLU) and self.use_meta_learning and task_trajectories is not None and not modulated_policy:
                if x_policy.shape[-1] == self.hidden_dim // 2:
                    x_policy = x_policy * gains_policy + biases_policy
                    modulated_policy = True
        policy_logits = x_policy

        # Value network with optional modulation
        x_value = combined_repr
        modulated_value = False
        for i, layer in enumerate(self.value_net):
            x_value = layer(x_value)
            if isinstance(layer, nn.ReLU) and self.use_meta_learning and task_trajectories is not None and not modulated_value:
                if x_value.shape[-1] == self.hidden_dim // 2:
                    x_value = x_value * gains_value + biases_value
                    modulated_value = True
        value = x_value

        # Apply action mask
        masked_logits = policy_logits
        if action_mask is not None:
            if action_mask.shape[0] == policy_logits.shape[0]:
                masked_logits = policy_logits.clone()
                masked_logits[~action_mask] = -1e9

        action_logits = masked_logits.nan_to_num(nan=-1e9, posinf=1e9, neginf=-1e9)

        return_dict: HypothesisNetOutput = {
            'action_logits': action_logits,
            'value': value,
            'tree_representation': combined_repr # Return the augmented representation
        }
        if self.use_meta_learning and task_embedding_vector is not None:
            return_dict['task_embedding'] = task_embedding_vector

        return return_dict

    def get_action(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        action_mask: Optional[torch.Tensor] = None,
        task_trajectories: Optional[torch.Tensor] = None,
        tree_structure: Optional[Dict[int, List[int]]] = None,
        ai_model_representation: Optional[torch.Tensor] = None,
        ai_model_type_idx: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Selects an action for AIHypothesisNet, passing through AI-specific inputs to forward pass.
        """
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        else:
            obs_tensor = obs

        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        # Call AIHypothesisNet's forward method with all specific arguments
        outputs = self.forward(
            obs_tensor,
            action_mask=action_mask,
            task_trajectories=task_trajectories,
            tree_structure=tree_structure,
            ai_model_representation=ai_model_representation,
            ai_model_type_idx=ai_model_type_idx
        )

        dist = Categorical(logits=outputs['action_logits'])
        action_tensor = torch.argmax(outputs['action_logits'], dim=-1) if deterministic else dist.sample()
        log_prob_tensor = dist.log_prob(action_tensor)
        value_tensor = outputs['value']

        action_val: int = action_tensor.item()
        log_prob_val: float = log_prob_tensor.item()
        value_val: float = value_tensor.squeeze().item()

        return action_val, log_prob_val, value_val


if __name__ == "__main__":
    # This __main__ block is for testing HypothesisNet and AIHypothesisNet components in isolation.
    # It assumes the necessary dependencies (SymbolicDiscoveryEnv, ProgressiveGrammar, Variable) are available.

    # Dummy setup for testing
    grammar = ProgressiveGrammar()
    variables = [Variable("x", 0, {}), Variable("v", 1, {})]
    # Simple dummy data for environment, adjust as needed for actual env behavior
    data = np.column_stack([np.random.randn(100), np.random.randn(100) * 2, np.random.randn(100)])

    # Ensure SymbolicDiscoveryEnv can provide 'tree_structure' in info dict from reset/step if TreeEncoder is used.
    # The 'provide_tree_structure' is a hypothetical flag for the environment.
    # In a real setup, SymbolicDiscoveryEnv should correctly produce this info based on its internal state.
    try:
        env = SymbolicDiscoveryEnv(grammar, data, variables, max_depth=5, max_complexity=10, provide_tree_structure=True)
    except TypeError: # Handle if provide_tree_structure is not a valid argument
        env = SymbolicDiscoveryEnv(grammar, data, variables, max_depth=5, max_complexity=10)
        print("Warning: SymbolicDiscoveryEnv does not support 'provide_tree_structure'. TreeEncoder might not function as expected.")


    obs_dim = env.observation_space.shape[0] # Assuming observation space is flat
    action_dim = env.action_space.n

    # Test HypothesisNet with Transformer encoder and Meta-learning enabled
    print("--- Testing HypothesisNet (Transformer, Meta-learning) ---")
    policy_transformer = HypothesisNet(obs_dim, action_dim, grammar=grammar, use_meta_learning=True, encoder_type='transformer')
    print(f"Policy (Transformer, Meta) params: {sum(p.numel() for p in policy_transformer.parameters())}")

    obs, info = safe_env_reset(env) # Reset environment and get initial observation and info
    obs_tensor = torch.FloatTensor(np.array(obs)).unsqueeze(0) # (1, obs_dim)
    action_mask_np = env.get_action_mask() # (action_dim,)
    action_mask_tensor = torch.BoolTensor(action_mask_np).unsqueeze(0) # (1, action_dim)
    
    # Dummy task trajectories for meta-learning. Shape needs to match expected input for task_encoder.
    # The node_feature_dim is used here as a placeholder for the feature dimension of each step in a trajectory.
    dummy_trajs = torch.randn(1, 3, 10, policy_transformer.node_feature_dim) # (batch_size, num_trajs, traj_len, feat_dim)

    # Test forward pass
    outputs_transformer_meta = policy_transformer.forward(obs_tensor, action_mask_tensor, task_trajectories=dummy_trajs)
    print("Outputs (HypothesisNet, Transformer, Meta, with trajectories):")
    for k, v_ in outputs_transformer_meta.items():
        print(f"  {k}: {v_.shape if isinstance(v_, torch.Tensor) else v_}")

    # Test get_action method
    action_val, log_prob_val, value_val = policy_transformer.get_action(
        obs_tensor, action_mask_tensor, task_trajectories=dummy_trajs, deterministic=False
    )
    print(f"HypothesisNet get_action output (action, log_prob, value): ({action_val}, {log_prob_val:.4f}, {value_val:.4f})")


    # Test HypothesisNet with TreeLSTM encoder and Meta-learning enabled
    print("\n--- Testing HypothesisNet (TreeLSTM, Meta-learning) ---")
    # TreeEncoder requires tree_structure. Ensure env.reset() and env.step() provide this.
    policy_tree = HypothesisNet(obs_dim, action_dim, grammar=grammar, use_meta_learning=True, encoder_type='treelstm')
    print(f"Policy (TreeLSTM, Meta) params: {sum(p.numel() for p in policy_tree.parameters())}")
    
    # Attempt to get tree_structure from env info, if available
    tree_structure_example = info.get('tree_structure')
    if tree_structure_example is None:
        print("Warning: 'tree_structure' not found in env info. TreeEncoder might not function correctly.")

    outputs_tree_meta = policy_tree.forward(obs_tensor, action_mask_tensor, task_trajectories=dummy_trajs, tree_structure=tree_structure_example)
    print("Outputs (HypothesisNet, TreeLSTM, Meta, with trajectories):")
    for k, v_ in outputs_tree_meta.items():
        print(f"  {k}: {v_.shape if isinstance(v_, torch.Tensor) else v_}")
    
    action_val_tree, log_prob_val_tree, value_val_tree = policy_tree.get_action(
        obs_tensor, action_mask_tensor, task_trajectories=dummy_trajs, tree_structure=tree_structure_example, deterministic=False
    )
    print(f"HypothesisNet (TreeLSTM) get_action output (action, log_prob, value): ({action_val_tree}, {log_prob_val_tree:.4f}, {value_val_tree:.4f})")


    # Test AIHypothesisNet with Transformer encoder and Meta-learning enabled
    print("\n--- Testing AIHypothesisNet (Transformer, Meta-learning) ---")
    ai_policy_transformer = AIHypothesisNet(
        obs_dim, action_dim, grammar=grammar,
        use_meta_learning=True, encoder_type='transformer',
        hidden_dim=256 # Example hidden_dim
    )
    print(f"AIPolicy (Transformer, Meta) params: {sum(p.numel() for p in ai_policy_transformer.parameters())}")

    # Dummy AI model related inputs for AIHypothesisNet
    dummy_ai_model_repr = torch.randn(1, 5, ai_policy_transformer.hidden_dim) # (batch, seq_len_model, hidden_dim)
    dummy_ai_model_type = torch.randint(0, 10, (1,), dtype=torch.long) # (batch_size,)

    outputs_ai_transformer_meta = ai_policy_transformer.forward(
        obs_tensor,
        action_mask_tensor,
        task_trajectories=dummy_trajs,
        ai_model_representation=dummy_ai_model_repr,
        ai_model_type_idx=dummy_ai_model_type
    )
    print("Outputs (AIHypothesisNet, Transformer, Meta, with AI inputs):")
    for k, v_ in outputs_ai_transformer_meta.items():
        print(f"  {k}: {v_.shape if isinstance(v_, torch.Tensor) else v_}")

    # Test get_action for AIHypothesisNet
    action_info_ai = ai_policy_transformer.get_action(
        obs_tensor,
        action_mask=action_mask_tensor,
        task_trajectories=dummy_trajs,
        ai_model_representation=dummy_ai_model_repr,
        ai_model_type_idx=dummy_ai_model_type
    )
    print(f"AIHypothesisNet get_action output (action, log_prob, value): ({action_info_ai[0]}, {action_info_ai[1]:.4f}, {action_info_ai[2]:.4f})")