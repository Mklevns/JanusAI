# JanusAI/memory/advanced_features.py
"""
Advanced features for the Janus Dual Memory System.
Includes memory consolidation, visualization, and export capabilities.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import zipfile
import io

from janus_ai.dual_memory_system import (

    DualMemorySystem, Discovery, IntermediateResult,
    EpisodicMemory, SharedMemory, EmbeddingGenerator
)


class MemoryConsolidator:
    """
    Consolidates and compresses old memories to save space while preserving important information.
    Implements sleep-like consolidation inspired by neuroscience.
    """
    
    def __init__(self, memory_system: DualMemorySystem):
        self.memory_system = memory_system
        self.consolidation_history = []
        
    def consolidate_old_memories(self, 
                               age_threshold_days: int = 30,
                               min_importance: float = 0.5) -> Dict[str, Any]:
        """Consolidate memories older than threshold"""
        
        episodic = self.memory_system.episodic
        now = datetime.now()
        
        # Find old memories
        old_memories = []
        for discovery in episodic.memories.values():
            age = (now - discovery.timestamp).days
            if age > age_threshold_days:
                importance = self._calculate_importance(discovery)
                if importance < min_importance:
                    old_memories.append((discovery, importance))
        
        # Sort by importance (lowest first)
        old_memories.sort(key=lambda x: x[1])
        
        # Consolidate similar memories
        consolidated = self._consolidate_similar(old_memories)
        
        # Create summary discoveries
        summary_discoveries = []
        for group in consolidated:
            summary = self._create_summary_discovery(group)
            if summary:
                summary_discoveries.append(summary)
        
        # Remove original memories and add summaries
        removed_count = 0
        for discovery, _ in old_memories:
            if discovery.id in episodic.memories:
                del episodic.memories[discovery.id]
                removed_count += 1
        
        # Add summaries
        for summary in summary_discoveries:
            episodic.add(summary)
        
        result = {
            'removed_count': removed_count,
            'summary_count': len(summary_discoveries),
            'space_saved_mb': self._estimate_space_saved(old_memories),
            'timestamp': now
        }
        
        self.consolidation_history.append(result)
        return result
    
    def _calculate_importance(self, discovery: Discovery) -> float:
        """Calculate importance score for a discovery"""
        
        # Factors:
        # - Validation score (40%)
        # - Confidence (20%)
        # - Uniqueness (20%)
        # - Access frequency (20%)
        
        validation_weight = discovery.validation_score * 0.4
        confidence_weight = discovery.confidence * 0.2
        
        # Uniqueness based on expression complexity
        uniqueness = min(1.0, len(discovery.expression) / 100) * 0.2
        
        # Access frequency (would need to track this)
        access_weight = 0.2  # Default
        
        return validation_weight + confidence_weight + uniqueness + access_weight
    
    def _consolidate_similar(self, 
                           memories: List[Tuple[Discovery, float]]) -> List[List[Discovery]]:
        """Group similar memories for consolidation"""
        
        if not memories:
            return []
        
        # Use embeddings if available
        discoveries_with_embeddings = [
            (d, imp) for d, imp in memories if d.embedding is not None
        ]
        
        if discoveries_with_embeddings:
            # Cluster based on embeddings
            embeddings = np.array([d[0].embedding for d in discoveries_with_embeddings])
            
            # DBSCAN clustering
            clustering = DBSCAN(eps=0.3, min_samples=2).fit(embeddings)
            
            # Group by cluster
            groups = defaultdict(list)
            for (discovery, _), label in zip(discoveries_with_embeddings, clustering.labels_):
                groups[label].append(discovery)
            
            return list(groups.values())
        else:
            # Fallback: group by domain
            groups = defaultdict(list)
            for discovery, _ in memories:
                groups[discovery.domain].append(discovery)
            
            return list(groups.values())
    
    def _create_summary_discovery(self, group: List[Discovery]) -> Optional[Discovery]:
        """Create a summary discovery from a group"""
        
        if not group:
            return None
        
        # Find the best representative
        best = max(group, key=lambda d: d.validation_score)
        
        # Create summary
        summary = Discovery(
            id=f"summary_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            domain=best.domain,
            expression=f"SUMMARY[{len(group)} discoveries]: {best.expression}",
            hypothesis=f"Consolidated from {len(group)} similar discoveries",
            evidence=[{
                'type': 'consolidation',
                'original_count': len(group),
                'date_range': f"{min(d.timestamp for d in group)} to {max(d.timestamp for d in group)}",
                'avg_validation': np.mean([d.validation_score for d in group])
            }],
            confidence=np.mean([d.confidence for d in group]),
            validation_score=np.mean([d.validation_score for d in group]),
            reasoning_trace=[f"Consolidated from: {d.expression}" for d in group[:3]],
            agent_roles=list(set(role for d in group for role in d.agent_roles)),
            metadata={
                'is_summary': True,
                'original_ids': [d.id for d in group]
            }
        )
        
        # Average embeddings if available
        embeddings = [d.embedding for d in group if d.embedding is not None]
        if embeddings:
            summary.embedding = np.mean(embeddings, axis=0)
        
        return summary
    
    def _estimate_space_saved(self, memories: List[Tuple[Discovery, float]]) -> float:
        """Estimate space saved in MB"""
        import sys
        
        total_size = sum(sys.getsizeof(d[0]) for d in memories)
        return total_size / (1024 * 1024)


class MemoryVisualizer:
    """Visualization tools for the memory system"""
    
    def __init__(self, memory_system: DualMemorySystem):
        self.memory_system = memory_system
    
    def plot_discovery_timeline(self, domain: Optional[str] = None, save_path: Optional[str] = None):
        """Plot timeline of discoveries"""
        
        episodic = self.memory_system.episodic
        discoveries = list(episodic.memories.values())
        
        if domain:
            discoveries = [d for d in discoveries if d.domain == domain]
        
        if not discoveries:
            print("No discoveries to plot")
            return
        
        # Prepare data
        df = pd.DataFrame({
            'timestamp': [d.timestamp for d in discoveries],
            'validation_score': [d.validation_score for d in discoveries],
            'confidence': [d.confidence for d in discoveries],
            'domain': [d.domain for d in discoveries]
        })
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot validation scores over time
        for domain in df['domain'].unique():
            domain_data = df[df['domain'] == domain]
            ax1.scatter(domain_data['timestamp'], domain_data['validation_score'], 
                       label=domain, alpha=0.6, s=50)
        
        ax1.set_ylabel('Validation Score')
        ax1.set_title('Discovery Quality Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot discovery rate
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        discovery_counts = df.groupby(['date', 'domain']).size().reset_index(name='count')
        
        for domain in discovery_counts['domain'].unique():
            domain_counts = discovery_counts[discovery_counts['domain'] == domain]
            ax2.plot(domain_counts['date'], domain_counts['count'], 
                    marker='o', label=domain)
        
        ax2.set_ylabel('Discoveries per Day')
        ax2.set_xlabel('Date')
        ax2.set_title('Discovery Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def plot_embedding_space(self, domain: Optional[str] = None, save_path: Optional[str] = None):
        """Visualize discoveries in embedding space using t-SNE"""
        
        episodic = self.memory_system.episodic
        
        # Get discoveries with embeddings
        discoveries = [
            d for d in episodic.memories.values() 
            if d.embedding is not None and (domain is None or d.domain == domain)
        ]
        
        if len(discoveries) < 2:
            print("Not enough discoveries with embeddings to visualize")
            return
        
        # Extract embeddings and metadata
        embeddings = np.array([d.embedding for d in discoveries])
        labels = [d.domain for d in discoveries]
        scores = [d.validation_score for d in discoveries]
        
        # Reduce dimensionality with t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        coords = tsne.fit_transform(embeddings)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot
        scatter = plt.scatter(coords[:, 0], coords[:, 1], 
                            c=scores, s=100, 
                            cmap='viridis', alpha=0.6, 
                            edgecolors='black', linewidth=1)
        
        # Add labels for high-scoring discoveries
        high_score_threshold = np.percentile(scores, 80)
        for i, (x, y) in enumerate(coords):
            if scores[i] > high_score_threshold:
                plt.annotate(f"{discoveries[i].expression[:20]}...", 
                           (x, y), fontsize=8, alpha=0.7)
        
        plt.colorbar(scatter, label='Validation Score')
        plt.title('Discovery Embedding Space (t-SNE)')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def plot_knowledge_graph(self, max_nodes: int = 50, save_path: Optional[str] = None):
        """Create a knowledge graph of discoveries and their relationships"""
        
        episodic = self.memory_system.episodic
        discoveries = list(episodic.memories.values())[:max_nodes]
        
        if not discoveries:
            print("No discoveries to visualize")
            return
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for discovery in discoveries:
            G.add_node(discovery.id, 
                      label=discovery.expression[:30],
                      domain=discovery.domain,
                      score=discovery.validation_score)
        
        # Add edges based on similarity
        for i, d1 in enumerate(discoveries):
            for d2 in discoveries[i+1:]:
                similarity = self._calculate_similarity(d1, d2)
                if similarity > 0.5:  # Threshold
                    G.add_edge(d1.id, d2.id, weight=similarity)
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create plot
        plt.figure(figsize=(15, 10))
        
        # Color by domain
        domains = list(set(d.domain for d in discoveries))
        colors = plt.cm.tab10(np.linspace(0, 1, len(domains)))
        domain_colors = dict(zip(domains, colors))
        
        node_colors = [domain_colors[G.nodes[node]['domain']] for node in G.nodes()]
        node_sizes = [G.nodes[node]['score'] * 1000 for node in G.nodes()]
        
        # Draw
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.7)
        nx.draw_networkx_edges(G, pos, alpha=0.3)
        
        # Add labels
        labels = {node: G.nodes[node]['label'] for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        # Legend
        for domain, color in domain_colors.items():
            plt.scatter([], [], c=[color], s=100, label=domain)
        plt.legend(title='Domain', loc='best')
        
        plt.title('Discovery Knowledge Graph')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def _calculate_similarity(self, d1: Discovery, d2: Discovery) -> float:
        """Calculate similarity between two discoveries"""
        
        # If embeddings available, use cosine similarity
        if d1.embedding is not None and d2.embedding is not None:
            cos_sim = np.dot(d1.embedding, d2.embedding) / (
                np.linalg.norm(d1.embedding) * np.linalg.norm(d2.embedding)
            )
            return float(cos_sim)
        
        # Fallback: simple text similarity
        tokens1 = set(d1.expression.split())
        tokens2 = set(d2.expression.split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union


class MemoryExporter:
    """Export and import memory for sharing discoveries"""
    
    def __init__(self, memory_system: DualMemorySystem):
        self.memory_system = memory_system
    
    def export_discoveries(self, 
                         output_path: str,
                         domain: Optional[str] = None,
                         min_score: float = 0.0) -> Dict[str, Any]:
        """Export discoveries to a shareable format"""
        
        episodic = self.memory_system.episodic
        
        # Filter discoveries
        discoveries = []
        for discovery in episodic.memories.values():
            if domain and discovery.domain != domain:
                continue
            if discovery.validation_score < min_score:
                continue
            discoveries.append(discovery)
        
        # Create export package
        export_data = {
            'metadata': {
                'export_date': datetime.now().isoformat(),
                'janus_version': '1.0.0',  # Would get from config
                'discovery_count': len(discoveries),
                'domains': list(set(d.domain for d in discoveries))
            },
            'discoveries': [d.to_dict() for d in discoveries]
        }
        
        # Create zip file with JSON and any embeddings
        with zipfile.ZipFile(output_path, 'w') as zf:
            # Main data
            zf.writestr('discoveries.json', json.dumps(export_data, indent=2))
            
            # Statistics
            stats = episodic.get_statistics()
            zf.writestr('statistics.json', json.dumps(stats, indent=2))
            
            # Embeddings (if any)
            embeddings = {}
            for d in discoveries:
                if d.embedding is not None:
                    embeddings[d.id] = d.embedding.tolist()
            
            if embeddings:
                import pickle
                zf.writestr('embeddings.pkl', pickle.dumps(embeddings))
        
        return {
            'exported_count': len(discoveries),
            'file_size_mb': Path(output_path).stat().st_size / (1024 * 1024),
            'domains': export_data['metadata']['domains']
        }
    
    def import_discoveries(self, import_path: str, 
                         merge_strategy: str = 'skip_existing') -> Dict[str, Any]:
        """Import discoveries from export file"""
        
        episodic = self.memory_system.episodic
        imported_count = 0
        skipped_count = 0
        
        with zipfile.ZipFile(import_path, 'r') as zf:
            # Load main data
            with zf.open('discoveries.json') as f:
                import_data = json.load(f)
            
            # Load embeddings if available
            embeddings = {}
            if 'embeddings.pkl' in zf.namelist():
                import pickle
                with zf.open('embeddings.pkl') as f:
                    embeddings = pickle.load(f)
            
            # Import discoveries
            for disc_data in import_data['discoveries']:
                discovery = Discovery.from_dict(disc_data)
                
                # Restore embedding if available
                if discovery.id in embeddings:
                    discovery.embedding = np.array(embeddings[discovery.id])
                
                # Handle merge strategy
                if discovery.id in episodic.memories:
                    if merge_strategy == 'skip_existing':
                        skipped_count += 1
                        continue
                    elif merge_strategy == 'replace':
                        pass  # Will be replaced
                    elif merge_strategy == 'keep_better':
                        existing = episodic.memories[discovery.id]
                        if existing.validation_score >= discovery.validation_score:
                            skipped_count += 1
                            continue
                
                # Add to memory
                episodic.add(discovery)
                imported_count += 1
        
        return {
            'imported_count': imported_count,
            'skipped_count': skipped_count,
            'total_in_file': len(import_data['discoveries']),
            'metadata': import_data['metadata']
        }


class ImportanceSampler:
    """Sample from memory based on importance scores"""
    
    def __init__(self, memory_system: DualMemorySystem):
        self.memory_system = memory_system
        
    def sample_by_importance(self, 
                           n_samples: int,
                           temperature: float = 1.0,
                           domain: Optional[str] = None) -> List[Discovery]:
        """Sample discoveries weighted by importance"""
        
        episodic = self.memory_system.episodic
        
        # Get candidates
        if domain:
            candidates = episodic.search_by_domain(domain, limit=1000)
        else:
            candidates = list(episodic.memories.values())
        
        if not candidates:
            return []
        
        # Calculate importance scores
        importance_scores = []
        for discovery in candidates:
            score = self._calculate_importance(discovery)
            importance_scores.append(score)
        
        # Convert to probabilities with temperature
        importance_scores = np.array(importance_scores)
        
        if temperature != 0:
            # Apply temperature scaling
            importance_scores = importance_scores / temperature
            # Softmax
            exp_scores = np.exp(importance_scores - np.max(importance_scores))
            probabilities = exp_scores / exp_scores.sum()
        else:
            # Greedy: select top-k
            indices = np.argsort(importance_scores)[-n_samples:]
            return [candidates[i] for i in indices]
        
        # Sample
     
