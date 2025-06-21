# JanusAI/memory/dual_memory_system.py
"""
Dual Memory System for Janus scientific discovery platform.
Implements both Episodic Memory (long-term) and Shared Memory (per-problem).
"""

import pickle
import sqlite3
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import threading
import hashlib
import faiss  # For efficient similarity search
import logging


@dataclass
class MemoryEntry:
    """Base class for memory entries"""
    id: str
    timestamp: datetime
    domain: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'domain': self.domain,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class Discovery(MemoryEntry):
    """Represents a scientific discovery for episodic memory"""
    expression: str
    hypothesis: str
    evidence: List[Dict[str, Any]]
    confidence: float
    validation_score: float
    reasoning_trace: List[str]
    agent_roles: List[str]
    embedding: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if not self.id:
            # Generate unique ID from content
            content = f"{self.expression}_{self.domain}_{self.timestamp}"
            self.id = hashlib.md5(content.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling numpy arrays"""
        data = super().to_dict()
        data.update({
            'expression': self.expression,
            'hypothesis': self.hypothesis,
            'evidence': self.evidence,
            'confidence': self.confidence,
            'validation_score': self.validation_score,
            'reasoning_trace': self.reasoning_trace,
            'agent_roles': self.agent_roles,
            'embedding': self.embedding.tolist() if self.embedding is not None else None
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Discovery':
        """Create from dictionary, handling numpy arrays"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('embedding'):
            data['embedding'] = np.array(data['embedding'])
        return cls(**data)


@dataclass 
class IntermediateResult(MemoryEntry):
    """Represents intermediate results for shared memory"""
    agent_role: str
    expression: str
    thought: str
    response: str
    score: float
    detailed_scores: Dict[str, float] = field(default_factory=dict)
    feedback: str = ""
    iteration: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = super().to_dict()
        data.update({
            'agent_role': self.agent_role,
            'expression': self.expression,
            'thought': self.thought,
            'response': self.response,
            'score': self.score,
            'detailed_scores': self.detailed_scores,
            'feedback': self.feedback,
            'iteration': self.iteration
        })
        return data


class MemoryIndex:
    """Efficient similarity index for memory retrieval"""
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance
        self.id_map = {}  # Maps index position to memory ID
        self.position_counter = 0
        
    def add(self, memory_id: str, embedding: np.ndarray):
        """Add embedding to index"""
        if embedding.shape[0] != self.dimension:
            raise ValueError(f"Embedding dimension {embedding.shape[0]} != {self.dimension}")
        
        self.index.add(embedding.reshape(1, -1))
        self.id_map[self.position_counter] = memory_id
        self.position_counter += 1
        
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors"""
        if self.index.ntotal == 0:
            return []
            
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx in self.id_map:
                results.append((self.id_map[idx], float(dist)))
        
        return results
    
    def remove(self, memory_id: str):
        """Remove from index (requires rebuilding)"""
        # FAISS doesn't support removal, so we'd need to rebuild
        # For now, we'll just remove from id_map
        positions_to_remove = [pos for pos, mid in self.id_map.items() if mid == memory_id]
        for pos in positions_to_remove:
            del self.id_map[pos]
    
    def save(self, path: str):
        """Save index to disk"""
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.map", 'wb') as f:
            pickle.dump((self.id_map, self.position_counter), f)
    
    def load(self, path: str):
        """Load index from disk"""
        self.index = faiss.read_index(f"{path}.index")
        with open(f"{path}.map", 'rb') as f:
            self.id_map, self.position_counter = pickle.load(f)


class EpisodicMemory:
    """
    Long-term memory for storing discoveries across problems.
    Supports similarity-based retrieval and persistent storage.
    """
    
    def __init__(self, 
                 capacity: int = 10000,
                 db_path: Optional[str] = None,
                 embedding_dim: int = 768):
        
        self.capacity = capacity
        self.db_path = db_path or "janus_episodic_memory.db"
        self.embedding_dim = embedding_dim
        
        # In-memory cache
        self.memories: Dict[str, Discovery] = {}
        self.domain_index: Dict[str, List[str]] = defaultdict(list)
        
        # Similarity index
        self.similarity_index = MemoryIndex(embedding_dim)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self._init_database()
        
        # Load existing memories
        self._load_from_database()
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS discoveries (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    expression TEXT NOT NULL,
                    hypothesis TEXT,
                    confidence REAL,
                    validation_score REAL,
                    data BLOB,
                    embedding BLOB
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_domain ON discoveries(domain)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_validation ON discoveries(validation_score)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp ON discoveries(timestamp)
            ''')
    
    def add(self, discovery: Discovery) -> bool:
        """Add a discovery to episodic memory"""
        with self.lock:
            # Check capacity
            if len(self.memories) >= self.capacity:
                self._evict_oldest()
            
            # Add to memory
            self.memories[discovery.id] = discovery
            self.domain_index[discovery.domain].append(discovery.id)
            
            # Add to similarity index if embedding available
            if discovery.embedding is not None:
                self.similarity_index.add(discovery.id, discovery.embedding)
            
            # Persist to database
            self._save_to_database(discovery)
            
            self.logger.info(f"Added discovery {discovery.id} to episodic memory")
            return True
    
    def get(self, discovery_id: str) -> Optional[Discovery]:
        """Retrieve a specific discovery"""
        with self.lock:
            return self.memories.get(discovery_id)
    
    def search_by_similarity(self, 
                           query_embedding: np.ndarray, 
                           k: int = 5,
                           domain: Optional[str] = None) -> List[Discovery]:
        """Search for similar discoveries using embeddings"""
        with self.lock:
            # Get candidates from similarity index
            candidates = self.similarity_index.search(query_embedding, k * 2)
            
            results = []
            for memory_id, distance in candidates:
                discovery = self.memories.get(memory_id)
                if discovery:
                    # Filter by domain if specified
                    if domain and discovery.domain != domain:
                        continue
                    results.append(discovery)
                    
                if len(results) >= k:
                    break
            
            return results
    
    def search_by_domain(self, domain: str, limit: int = 10) -> List[Discovery]:
        """Get all discoveries in a domain"""
        with self.lock:
            discovery_ids = self.domain_index.get(domain, [])[-limit:]
            return [self.memories[did] for did in discovery_ids if did in self.memories]
    
    def search_by_expression(self, pattern: str, limit: int = 10) -> List[Discovery]:
        """Search discoveries by expression pattern"""
        with self.lock:
            results = []
            for discovery in self.memories.values():
                if pattern.lower() in discovery.expression.lower():
                    results.append(discovery)
                    if len(results) >= limit:
                        break
            return results
    
    def get_top_validated(self, n: int = 10, domain: Optional[str] = None) -> List[Discovery]:
        """Get top discoveries by validation score"""
        with self.lock:
            discoveries = list(self.memories.values())
            
            if domain:
                discoveries = [d for d in discoveries if d.domain == domain]
            
            discoveries.sort(key=lambda d: d.validation_score, reverse=True)
            return discoveries[:n]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics"""
        with self.lock:
            stats = {
                'total_discoveries': len(self.memories),
                'domains': {},
                'average_confidence': 0.0,
                'average_validation': 0.0,
                'memory_usage_mb': self._estimate_memory_usage() / 1024 / 1024
            }
            
            # Domain breakdown
            for domain, ids in self.domain_index.items():
                stats['domains'][domain] = len(ids)
            
            # Averages
            if self.memories:
                confidences = [d.confidence for d in self.memories.values()]
                validations = [d.validation_score for d in self.memories.values()]
                stats['average_confidence'] = np.mean(confidences)
                stats['average_validation'] = np.mean(validations)
            
            return stats
    
    def _evict_oldest(self):
        """Evict oldest discovery when at capacity"""
        if not self.memories:
            return
            
        # Find oldest by timestamp
        oldest = min(self.memories.values(), key=lambda d: d.timestamp)
        
        # Remove from all indices
        del self.memories[oldest.id]
        self.domain_index[oldest.domain].remove(oldest.id)
        self.similarity_index.remove(oldest.id)
        
        # Remove from database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM discoveries WHERE id = ?", (oldest.id,))
        
        self.logger.info(f"Evicted oldest discovery {oldest.id}")
    
    def _save_to_database(self, discovery: Discovery):
        """Save discovery to database"""
        with sqlite3.connect(self.db_path) as conn:
            # Serialize complex fields
            data_blob = pickle.dumps({
                'evidence': discovery.evidence,
                'reasoning_trace': discovery.reasoning_trace,
                'agent_roles': discovery.agent_roles,
                'metadata': discovery.metadata
            })
            
            embedding_blob = pickle.dumps(discovery.embedding) if discovery.embedding is not None else None
            
            conn.execute('''
                INSERT OR REPLACE INTO discoveries 
                (id, timestamp, domain, expression, hypothesis, confidence, validation_score, data, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                discovery.id,
                discovery.timestamp.isoformat(),
                discovery.domain,
                discovery.expression,
                discovery.hypothesis,
                discovery.confidence,
                discovery.validation_score,
                data_blob,
                embedding_blob
            ))
    
    def _load_from_database(self):
        """Load all discoveries from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT id, timestamp, domain, expression, hypothesis, 
                       confidence, validation_score, data, embedding
                FROM discoveries
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (self.capacity,))
            
            for row in cursor:
                # Deserialize complex fields
                data = pickle.loads(row[7])
                embedding = pickle.loads(row[8]) if row[8] else None
                
                discovery = Discovery(
                    id=row[0],
                    timestamp=datetime.fromisoformat(row[1]),
                    domain=row[2],
                    expression=row[3],
                    hypothesis=row[4],
                    confidence=row[5],
                    validation_score=row[6],
                    evidence=data['evidence'],
                    reasoning_trace=data['reasoning_trace'],
                    agent_roles=data['agent_roles'],
                    metadata=data.get('metadata', {}),
                    embedding=embedding
                )
                
                self.memories[discovery.id] = discovery
                self.domain_index[discovery.domain].append(discovery.id)
                
                if embedding is not None:
                    self.similarity_index.add(discovery.id, embedding)
        
        self.logger.info(f"Loaded {len(self.memories)} discoveries from database")
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage in bytes"""
        # Rough estimation
        return len(pickle.dumps(self.memories))
    
    def save_index(self, path: Optional[str] = None):
        """Save similarity index to disk"""
        index_path = path or f"{self.db_path}.faiss"
        self.similarity_index.save(index_path)
    
    def load_index(self, path: Optional[str] = None):
        """Load similarity index from disk"""
        index_path = path or f"{self.db_path}.faiss"
        if Path(f"{index_path}.index").exists():
            self.similarity_index.load(index_path)


class SharedMemory:
    """
    Per-problem shared memory for intermediate results.
    Used by agents during a single discovery session.
    """
    
    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.results: List[IntermediateResult] = []
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Track access patterns
        self.access_count = defaultdict(int)
        self.last_accessed = {}
    
    def add(self, result: IntermediateResult) -> bool:
        """Add intermediate result to shared memory"""
        with self.lock:
            # Check if we need to make room
            if len(self.results) >= self.capacity:
                # Remove lowest scoring result
                self.results.sort(key=lambda r: r.score)
                removed = self.results.pop(0)
                self.logger.debug(f"Evicted result from {removed.agent_role} (score: {removed.score})")
            
            # Add new result
            self.results.append(result)
            
            # Sort by score (descending)
            self.results.sort(key=lambda r: r.score, reverse=True)
            
            self.logger.debug(f"Added result from {result.agent_role} (score: {result.score})")
            return True
    
    def get_top(self, k: int = 1) -> List[IntermediateResult]:
        """Get top k results by score"""
        with self.lock:
            self.access_count['get_top'] += 1
            return self.results[:k].copy()
    
    def get_by_agent(self, agent_role: str) -> List[IntermediateResult]:
        """Get all results from a specific agent"""
        with self.lock:
            self.access_count[f'agent_{agent_role}'] += 1
            return [r for r in self.results if r.agent_role == agent_role]
    
    def get_by_iteration(self, iteration: int) -> List[IntermediateResult]:
        """Get all results from a specific iteration"""
        with self.lock:
            return [r for r in self.results if r.iteration == iteration]
    
    def get_unique_expressions(self) -> List[str]:
        """Get all unique expressions in memory"""
        with self.lock:
            seen = set()
            unique = []
            for result in self.results:
                if result.expression not in seen:
                    seen.add(result.expression)
                    unique.append(result.expression)
            return unique
    
    def update_scores(self, updates: Dict[str, float]):
        """Update scores for specific results"""
        with self.lock:
            for result in self.results:
                if result.id in updates:
                    result.score = updates[result.id]
            
            # Re-sort by score
            self.results.sort(key=lambda r: r.score, reverse=True)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics"""
        with self.lock:
            if not self.results:
                return {
                    'num_results': 0,
                    'average_score': 0.0,
                    'best_score': 0.0,
                    'agents': [],
                    'iterations': []
                }
            
            scores = [r.score for r in self.results]
            agents = list(set(r.agent_role for r in self.results))
            iterations = list(set(r.iteration for r in self.results))
            
            return {
                'num_results': len(self.results),
                'average_score': np.mean(scores),
                'best_score': max(scores),
                'worst_score': min(scores),
                'score_std': np.std(scores),
                'agents': agents,
                'iterations': sorted(iterations),
                'access_patterns': dict(self.access_count)
            }
    
    def clear(self):
        """Clear all results"""
    
