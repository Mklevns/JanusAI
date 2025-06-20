# src/janus/ml/networks/encoders.py
"""
Feature encoder classes for Janus ML modules.
Includes multiple Tree encoders (LSTM-based and TreeLSTM-based), Transformer encoder,
and hybrid encoder combining tree and sequence modalities.
"""
import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# If TreeLSTMCell is available in your codebase, import it; else define or remove TreeLSTMEncoder
try:
    from janus.ml.networks.tree_lstm import TreeLSTMCell
except ImportError:
    TreeLSTMCell = None


class BaseEncoder(nn.Module, ABC):
    """Abstract base class for all encoders."""
    @abstractmethod
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass

    @abstractmethod
    def get_output_dim(self) -> int:
        pass


class RNNTreeEncoder(BaseEncoder):
    """
    Encodes tree-structured expressions via pre-order traversal sequences using LSTM.
    Supports mean, max, last, and attention pooling. Optionally adds depth encoding.
    """
    def __init__(
        self,
        vocab_size: int,
        node_embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        pooling: str = 'mean',
        dropout: float = 0.1,
        bidirectional: bool = True
    ):
        super().__init__()
        self.pooling = pooling
        self.bidirectional = bidirectional
        self.node_embedding = nn.Embedding(vocab_size, node_embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(
            input_size=node_embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )
        rnn_out = hidden_dim * (2 if bidirectional else 1)
        if pooling == 'attention':
            self.attn = nn.MultiheadAttention(
                embed_dim=rnn_out, num_heads=8, dropout=dropout, batch_first=True
            )
            self.attn_q = nn.Parameter(torch.randn(1,1,rnn_out))
        self.proj = nn.Sequential(
            nn.Linear(rnn_out, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self._output_dim = hidden_dim

    def forward(
        self,
        node_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        node_depths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        bs, seq = node_ids.size()
        if mask is None:
            mask = (node_ids != 0).float()
        x = self.node_embedding(node_ids)
        if node_depths is not None:
            x = x + self._encode_depths(node_depths)
        lengths = mask.sum(dim=1).long()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out_p, (h,c) = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_p, batch_first=True)
        if self.pooling == 'mean':
            pooled = (out * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        elif self.pooling == 'max':
            out = out.masked_fill(mask.unsqueeze(-1)==0, float('-inf'))
            pooled,_ = out.max(dim=1)
        elif self.pooling == 'last':
            if self.bidirectional:
                pooled = torch.cat([h[-2],h[-1]], dim=-1)
            else:
                pooled = h[-1]
        elif self.pooling == 'attention':
            q = self.attn_q.expand(bs, -1, -1)
            pooled, _ = self.attn(q, out, out, key_padding_mask=(mask==0))
            pooled = pooled.squeeze(1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        return self.proj(pooled)

    def _encode_depths(self, depths: torch.Tensor) -> torch.Tensor:
        bs, seq = depths.size()
        d = self.node_embedding.embedding_dim
        pe = torch.zeros(bs, seq, d, device=depths.device)
        div = torch.exp(torch.arange(0,d,2,device=depths.device)*(-math.log(10000)/d))
        pe[:,:,0::2] = torch.sin(depths.unsqueeze(-1)*div)
        pe[:,:,1::2] = torch.cos(depths.unsqueeze(-1)*div)
        return pe

    def get_output_dim(self) -> int:
        return self._output_dim


class TreeLSTMEncoder(BaseEncoder):
    """
    Encodes tree-structured inputs using a TreeLSTMCell for true tree aggregation.
    Requires TreeLSTMCell in the codebase.
    """
    def __init__(self, node_feature_dim: int, hidden_dim: int, n_layers: int = 1):
        super().__init__()
        if TreeLSTMCell is None:
            raise ImportError("TreeLSTMCell not available; install or implement it.")
        self.embed = nn.Linear(node_feature_dim, hidden_dim)
        self.cells = nn.ModuleList([TreeLSTMCell(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self._output_dim = hidden_dim

    def forward(self, x: torch.Tensor, tree_structure: Optional[Dict[int,List[int]]] = None) -> torch.Tensor:
        # x: [batch, max_nodes, node_dim]
        bs, max_nodes, _ = x.size()
        h_prev = self.embed(x)
        for layer, cell in enumerate(self.cells):
            memo = {}
            def recurse(node_idx: int) -> Any:
                if (layer,node_idx) in memo:
                    return memo[(layer,node_idx)]
                children = tree_structure.get(node_idx, []) if tree_structure else []
                hs,cs = [],[]
                for c in children:
                    h_c,c_c = recurse(c)
                    hs.append(h_c); cs.append(c_c)
                inp = h_prev[:,node_idx,:] if layer==0 else memo[(layer-1,node_idx)][0]
                h,c = cell(inp, hs, cs)
                memo[(layer,node_idx)] = (h,c)
                return h,c
            recurse(0)
            # collect new h_prev
            h_prev = torch.stack([memo[(layer,i)][0] for i in range(max_nodes)], dim=1)
        return h_prev[:,0,:]

    def get_output_dim(self) -> int:
        return self._output_dim


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model: int, dropout: float=0.1, max_len: int=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0,max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0,d_model,2)*( -math.log(10000)/d_model))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:,:x.size(1),:]
        return self.dropout(x)


class TransformerEncoder(BaseEncoder):
    """
    Transformer encoder wrapper with pooling options: cls, mean, max.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        pooling: str = 'cls',
        add_positional_encoding: bool = True
    ):
        super().__init__()
        self.pooling = pooling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        self.add_pe = add_positional_encoding
        if add_positional_encoding:
            self.pe = PositionalEncoding(d_model, dropout, max_seq_len)
        if pooling=='cls':
            self.cls_token = nn.Parameter(torch.randn(1,1,d_model))
        self._output_dim = d_model

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        return_sequence: bool = False
    ) -> torch.Tensor:
        bs, seq, _ = x.size()
        if self.pooling=='cls':
            cls = self.cls_token.expand(bs,-1,-1)
            x = torch.cat([cls,x],dim=1)
            if src_key_padding_mask is not None:
                mask = torch.zeros(bs,1,device=x.device,dtype=torch.bool)
                src_key_padding_mask = torch.cat([mask,src_key_padding_mask],dim=1)
        if self.add_pe:
            x = self.pe(x)
        x = self.layer_norm(x)
        out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        if return_sequence:
            return out
        if self.pooling=='cls':
            return out[:,0]
        mask = (~src_key_padding_mask).float().unsqueeze(-1) if src_key_padding_mask is not None else None
        if self.pooling=='mean':
            if mask is not None:
                return (out*mask).sum(1)/mask.sum(1)
            return out.mean(1)
        if self.pooling=='max':
            if mask is not None:
                out = out.masked_fill(src_key_padding_mask.unsqueeze(-1), float('-inf'))
            return out.max(1)[0]
        raise ValueError(f"Unknown pooling: {self.pooling}")

    def get_output_dim(self) -> int:
        return self._output_dim


class HybridEncoder(BaseEncoder):
    """
    Combines RNNTreeEncoder (or TreeLSTMEncoder) and TransformerEncoder.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        tree_hidden_dim: int = 128,
        transformer_dim: int = 128,
        num_transformer_layers: int = 4,
        combination: str = 'concat'
    ):
        super().__init__()
        self.combination = combination
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.tree_enc = RNNTreeEncoder(vocab_size, embed_dim, tree_hidden_dim, pooling='attention')
        self.trans_enc = TransformerEncoder(transformer_dim, pooling='mean', d_model=transformer_dim)
        if combination=='concat':
            self._output_dim = tree_hidden_dim + transformer_dim
            self.proj = nn.Linear(self._output_dim, self._output_dim)
        elif combination=='add':
            assert tree_hidden_dim==transformer_dim
            self._output_dim = tree_hidden_dim
            self.proj = nn.Identity()
        elif combination=='gate':
            assert tree_hidden_dim==transformer_dim
            self._output_dim = tree_hidden_dim
            self.gate = nn.Sequential(nn.Linear(tree_hidden_dim*2, tree_hidden_dim), nn.Sigmoid())
        else:
            raise ValueError

    def forward(self, node_ids: torch.Tensor, mask: Optional[torch.Tensor]=None, depths: Optional[torch.Tensor]=None) -> torch.Tensor:
        tree_feat = self.tree_enc(node_ids, mask, depths)
        seq = self.embedding(node_ids)
        mask_pad = None if mask is None else (mask==0)
        seq_feat = self.trans_enc(seq, mask_pad)
        if self.combination=='concat':
            out = torch.cat([tree_feat, seq_feat],dim=-1)
            return self.proj(out)
        if self.combination=='add':
            return tree_feat + seq_feat
        gate = self.gate(torch.cat([tree_feat, seq_feat],dim=-1))
        return gate*tree_feat + (1-gate)*seq_feat

    def get_output_dim(self) -> int:
        return self._output_dim


class AttentionPooling(nn.Module):
    """Learnable attention pooling for sequence-to-vector."""
    def __init__(self, input_dim: int, hidden_dim: Optional[int]=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        self.attn = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim,1))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        w = self.attn(x).squeeze(-1)
        if mask is not None:
            w = w.masked_fill(mask==0, float('-inf'))
        w = F.softmax(w, dim=-1).unsqueeze(-1)
        return torch.sum(w * x, dim=1)


def create_encoder(encoder_type: str, **kwargs) -> BaseEncoder:
    t = encoder_type.lower()
    if t=='rnn_tree': return RNNTreeEncoder(**kwargs)
    if t=='treelstm': return TreeLSTMEncoder(**kwargs)
    if t=='transformer': return TransformerEncoder(**kwargs)
    if t=='hybrid': return HybridEncoder(**kwargs)
    raise ValueError(f"Unknown encoder: {encoder_type}")

# --- Integration Wrappers ---
from enhanced_integration import (
    SymbolicExpressionHandler,
    EnhancedSymbolicRegressor,
    EnhancedGPT2AttentionExperiment,
)

# Expose them at module-level
__all__ += [
    "SymbolicExpressionHandler",
    "EnhancedSymbolicRegressor",
    "EnhancedGPT2AttentionExperiment",
]
