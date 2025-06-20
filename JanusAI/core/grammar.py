# src/janus/core/grammar.py
"""
AIGrammar: Core grammar for symbolic expression generation with transformer attention primitives.
Includes both symbolic (SymPy) and numerical implementations for attention operations.
"""

import numpy as np
import torch
import sympy as sp
from typing import Any, Dict, List, Optional, Callable
import torch.nn.functional as F


class AttentionPrimitives:
    """
    Attention-specific operations usable in both symbolic and numerical modes.
    """
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def attention_score(self, query: Any, key: Any, scale: Optional[float] = None) -> Any:
        # Symbolic branch
        if self._is_symbolic(query, key):
            return sp.Function('AttentionScore')(query, key)
        # Numerical branch
        q = torch.tensor(query, dtype=torch.float32, device=self.device) if isinstance(query, (list, np.ndarray)) else query.to(self.device)
        k = torch.tensor(key,   dtype=torch.float32, device=self.device) if isinstance(key,   (list, np.ndarray)) else key.to(self.device)
        if q.dim() == 1: q = q.unsqueeze(0)
        if k.dim() == 1: k = k.unsqueeze(0)
        scores = q @ k.transpose(-2, -1)
        if scale is None:
            scale = 1.0 / np.sqrt(q.shape[-1])
        return (scores * scale).cpu().numpy()

    def softmax_attention(self, scores: Any, mask: Optional[Any] = None) -> Any:
        if self._is_symbolic(scores):
            return sp.Function('Softmax')(scores)
        t = torch.tensor(scores, dtype=torch.float32, device=self.device) if isinstance(scores, (list, np.ndarray)) else scores.to(self.device)
        if mask is not None:
            m = torch.tensor(mask, dtype=torch.bool, device=self.device) if isinstance(mask, (list, np.ndarray)) else mask.to(self.device)
            t = t.masked_fill(~m, float('-inf'))
        return F.softmax(t, dim=-1).cpu().numpy()

    def weighted_value(self, weights: Any, values: Any) -> Any:
        if self._is_symbolic(weights, values):
            return sp.Function('WeightedSum')(weights, values)
        w = torch.tensor(weights, dtype=torch.float32, device=self.device) if isinstance(weights, (list, np.ndarray)) else weights.to(self.device)
        v = torch.tensor(values,  dtype=torch.float32, device=self.device) if isinstance(values, (list, np.ndarray)) else values.to(self.device)
        # matmul if 2D else elementwise
        if w.dim() == 2 and v.dim() == 2:
            out = w @ v
        else:
            out = w.unsqueeze(-1) * v
        return out.cpu().numpy()

    def position_encoding(self, positions: Any, d_model: int = 512) -> Any:
        if self._is_symbolic(positions):
            return sp.Function('PosEnc')(positions)
        pos = np.array(positions, dtype=int) if not isinstance(positions, np.ndarray) else positions
        enc = np.zeros((len(pos), d_model), float)
        for idx, p in enumerate(pos):
            for i in range(0, d_model, 2):
                angle = p / (10000 ** (2 * i / d_model))
                enc[idx, i]     = np.sin(angle)
                enc[idx, i + 1] = np.cos(angle)
        return enc

    def causal_mask(self, seq_len: int) -> np.ndarray:
        mask = np.tril(np.ones((seq_len, seq_len), bool), k=0)
        return mask

    def attention_pattern_detector(self, weights: Any, pattern: str = 'previous_token') -> Any:
        if self._is_symbolic(weights):
            return sp.Function(f'Pattern_{pattern}')(weights)
        arr = np.array(weights)
        if pattern == 'previous_token':
            out = np.zeros_like(arr)
            for i in range(1, arr.shape[0]):
                out[i, i-1] = arr[i, i-1]
            return out
        elif pattern == 'copying':
            m = min(arr.shape)
            out = np.zeros_like(arr)
            for i in range(m): out[i, i] = arr[i, i]
            return out
        # default zero
        return np.zeros_like(arr)

    def _is_symbolic(self, *args) -> bool:
        return any(isinstance(a, (sp.Expr, sp.Symbol)) for a in args)


class AIGrammar:
    """
    Core grammar for symbolic regression, extended with attention primitives.
    """
    def __init__(self, base_grammar: Any = None):
        self.base = base_grammar
        self.attn = AttentionPrimitives()
        # primitive categories: attention, positional, pattern, custom
        self.primitives: Dict[str, Dict[str, Callable]] = {
            'attention': {}, 'position': {}, 'pattern': {}, 'custom': {}
        }
        self._register_attention_primitives()

    def _register_attention_primitives(self):
        # attention ops
        self.primitives['attention'].update({
            'attention_score': self.attn.attention_score,
            'softmax_attention': self.attn.softmax_attention,
            'weighted_value': self.attn.weighted_value,
        })
        # position ops
        self.primitives['position'].update({
            'position_encoding': self.attn.position_encoding,
            'causal_mask': self.attn.causal_mask,
        })
        # pattern ops
        self.primitives['pattern'].update({
            'previous_token': lambda w: self.attn.attention_pattern_detector(w, 'previous_token'),
            'copying':      lambda w: self.attn.attention_pattern_detector(w, 'copying'),
        })
        # custom ops (example entropy)
        self.primitives['custom'].update({
            'attention_entropy': self._attention_entropy,
            'attention_sparsity': self._attention_sparsity,
        })

    def _attention_entropy(self, weights: Any) -> Any:
        if self.attn._is_symbolic(weights):
            return sp.Function('AttentionEntropy')(weights)
        arr = np.array(weights) + 1e-8
        ent = -np.sum(arr * np.log(arr), axis=-1)
        return ent

    def _attention_sparsity(self, weights: Any, thr: float = 0.1) -> Any:
        if self.attn._is_symbolic(weights):
            return sp.Function('AttentionSparsity')(weights)
        arr = np.array(weights)
        return np.mean(arr > thr, axis=-1)

    def get_primitive(self, name: str) -> Optional[Callable]:
        for cat in self.primitives.values():
            if name in cat:
                return cat[name]
        return None

    def list_primitives(self) -> Dict[str, List[str]]:
        return {cat: list(ops.keys()) for cat, ops in self.primitives.items()}


# Testing symbolic vs numerical behavior
def _test():
    import sympy as sp
    g = AIGrammar()
    # symbolic
    q, k, v = sp.symbols('q k v')
    print(g.attn.attention_score(q, k))
    print(g.attn.softmax_attention(sp.Function('S')(q)))
    # numerical
    Q = np.random.randn(2,4); K = np.random.randn(2,4); V = np.random.randn(2,4)
    scores = g.attn.attention_score(Q, K)
    w = g.attn.softmax_attention(scores)
    out = g.attn.weighted_value(w, V)
    print('Shapes:', scores.shape, w.shape, out.shape)
    # list
    print(g.list_primitives())

if __name__ == '__main__':
    _test()
