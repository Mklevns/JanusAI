# JanusAI/tests/test_memory_system.py
"""
Comprehensive test suite for the Janus Dual Memory System.
Tests all major functionality including thread safety and persistence.
"""

import unittest
import tempfile
import shutil
import numpy as np
import time
from datetime import datetime, timedelta
import threading
from pathlib import Path

from janus_ai.memory.dual_memory_system import (
    DualMemorySystem, EpisodicMemory, SharedMemory,
    Discovery, IntermediateResult, EmbeddingGenerator
)
from janus_ai.memory.memory_integration import (
    MemoryIntegratedEnv, MemoryAugmentedAgent,
    MemoryReplayBuffer, MemoryMetrics
)
from janus_ai.memory.advanced_features import (
    MemoryConsolidator, MemoryVisualizer,
    MemoryExporter, ImportanceSampler
)


class TestEpisodicMemory(unittest.TestCase):
    """Test episodic memory functionality"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_episodic.db"
        self.episodic = EpisodicMemory(
            capacity=100,
            db_path=str(self.db_path),
            embedding_dim=64  # Smaller for tests
        )
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_add_and_retrieve(self):
        """Test adding and retrieving discoveries"""
        discovery = self._create_test_discovery("test_1")
        
        # Add discovery
        success = self.episodic.add(discovery)
        self.assertTrue(success)
        
        # Retrieve by ID
        retrieved = self.episodic.get(discovery.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.expression, discovery.expression)
    
    def test_capacity_limit(self):
        """Test that old discoveries are evicted at capacity"""
        # Fill to capacity
        for i in range(105):  # Over capacity
            discovery = self._create_test_discovery(f"test_{i}")
            self.episodic.add(discovery)
        
        # Check capacity is maintained
        self.assertLessEqual(len(self.episodic.memories), 100)
        
        # Check oldest were evicted (first 5 should be gone)
        for i in range(5):
            self.assertIsNone(self.episodic.get(f"test_{i}"))
    
    def test_search_by_domain(self):
        """Test domain-based search"""
        # Add discoveries in different domains
        for i in range(10):
            domain = "attention" if i % 2 == 0 else "physics"
            discovery = self._create_test_discovery(f"test_{i}", domain=domain)
            self.episodic.add(discovery)
        
        # Search by domain
        attention_discoveries = self.episodic.search_by_domain("attention", limit=10)
        self.assertEqual(len(attention_discoveries), 5)
        
        for discovery in attention_discoveries:
            self.assertEqual(discovery.domain, "attention")
    
    def test_similarity_search(self):
        """Test embedding-based similarity search"""
        # Add discoveries with embeddings
        base_embedding = np.random.randn(64).astype(np.float32)
        
        for i in range(5):
            discovery = self._create_test_discovery(f"similar_{i}")
            # Create similar embeddings
            discovery.embedding = base_embedding + np.random.randn(64) * 0.1
            self.episodic.add(discovery)
        
        # Add different discoveries
        for i in range(5):
            discovery = self._create_test_discovery(f"different_{i}")
            discovery.embedding = np.random.randn(64).astype(np.float32)
            self.episodic.add(discovery)
        
        # Search for similar
        results = self.episodic.search_by_similarity(base_embedding, k=5)
        
        # Should find the similar ones
        similar_ids = [r.id for r in results]
        for i in range(5):
            self.assertIn(f"similar_{i}", similar_ids)
    
    def test_persistence(self):
        """Test database persistence"""
        # Add discoveries
        for i in range(5):
            discovery = self._create_test_discovery(f"persist_{i}")
            self.episodic.add(discovery)
        
        # Create new episodic memory with same DB
        new_episodic = EpisodicMemory(
            capacity=100,
            db_path=str(self.db_path),
            embedding_dim=64
        )
        
        # Check discoveries are loaded
        self.assertEqual(len(new_episodic.memories), 5)
        
        for i in range(5):
            retrieved = new_episodic.get(f"persist_{i}")
            self.assertIsNotNone(retrieved)
    
    def test_statistics(self):
        """Test statistics calculation"""
        # Add discoveries with known scores
        for i in range(10):
            discovery = self._create_test_discovery(f"stats_{i}")
            discovery.confidence = 0.5 + i * 0.05
            discovery.validation_score = 0.6 + i * 0.04
            self.episodic.add(discovery)
        
        stats = self.episodic.get_statistics()
        
        self.assertEqual(stats['total_discoveries'], 10)
        self.assertAlmostEqual(stats['average_confidence'], 0.725, places=2)
        self.assertAlmostEqual(stats['average_validation'], 0.78, places=2)
    
    def _create_test_discovery(self, id_suffix: str, domain: str = "test") -> Discovery:
        """Helper to create test discoveries"""
        return Discovery(
            id=id_suffix,
            timestamp=datetime.now(),
            domain=domain,
            expression=f"expression_{id_suffix}",
            hypothesis=f"hypothesis_{id_suffix}",
            evidence=[{"test": True}],
            confidence=0.8,
            validation_score=0.85,
            reasoning_trace=["step1", "step2"],
            agent_roles=["TestAgent"]
        )


class TestSharedMemory(unittest.TestCase):
    """Test shared memory functionality"""
    
    def setUp(self):
        self.shared = SharedMemory(capacity=5)
    
    def test_add_and_ranking(self):
        """Test adding results and automatic ranking"""
        # Add results with different scores
        for i in range(5):
            result = self._create_test_result(f"result_{i}", score=i * 0.2)
            self.shared.add(result)
        
        # Check top results are highest scoring
        top_results = self.shared.get_top(3)
        self.assertEqual(len(top_results), 3)
        
        # Should be in descending order
        for i in range(2):
            self.assertGreater(top_results[i].score, top_results[i+1].score)
    
    def test_capacity_eviction(self):
        """Test that lowest scoring results are evicted"""
        # Fill capacity
        for i in range(5):
            result = self._create_test_result(f"result_{i}", score=i * 0.2)
            self.shared.add(result)
        
        # Add one more with medium score
        new_result = self._create_test_result("new_result", score=0.5)
        self.shared.add(new_result)
        
        # Check capacity maintained
        self.assertEqual(len(self.shared.results), 5)
        
        # Check lowest score was evicted
        all_scores = [r.score for r in self.shared.results]
        self.assertNotIn(0.0, all_scores)  # Lowest score removed
        self.assertIn(0.5, all_scores)     # New score added
    
    def test_get_by_agent(self):
        """Test filtering by agent role"""
        # Add results from different agents
        agents = ["Agent1", "Agent2", "Agent3"]
        for i, agent in enumerate(agents):
            for j in range(2):
                result = self._create_test_result(f"{agent}_{j}", agent_role=agent)
                self.shared.add(result)
        
        # Get by specific agent
        agent1_results = self.shared.get_by_agent("Agent1")
        self.assertEqual(len(agent1_results), 2)
        
        for result in agent1_results:
            self.assertEqual(result.agent_role, "Agent1")
    
    def test_unique_expressions(self):
        """Test getting unique expressions"""
        # Add results with some duplicate expressions
        expressions = ["expr1", "expr2", "expr1", "expr3", "expr2"]
        for i, expr in enumerate(expressions):
            result = self._create_test_result(f"result_{i}")
            result.expression = expr
            self.shared.add(result)
        
        unique = self.shared.get_unique_expressions()
        self.assertEqual(len(unique), 3)
        self.assertIn("expr1", unique)
        self.assertIn("expr2", unique)
        self.assertIn("expr3", unique)
    
    def test_thread_safety(self):
        """Test concurrent access to shared memory"""
        errors = []
        
        def add_results(thread_id: int):
            try:
                for i in range(20):
                    result = self._create_test_result(
                        f"thread_{thread_id}_result_{i}",
                        score=np.random.random()
                    )
                    self.shared.add(result)
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=add_results, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check no errors
        self.assertEqual(len(errors), 0)
        
        # Check memory is in valid state
        self.assertLessEqual(len(self.shared.results), self.shared.capacity)
        
        # Check results are properly sorted
        scores = [r.score for r in self.shared.results]
        self.assertEqual(scores, sorted(scores, reverse=True))
    
    def _create_test_result(self, id_suffix: str, 
                          score: float = 0.5,
                          agent_role: str = "TestAgent") -> IntermediateResult:
        """Helper to create test results"""
        return IntermediateResult(
            id=id_suffix,
            timestamp=datetime.now(),
            domain="test",
            agent_role=agent_role,
            expression=f"expr_{id_suffix}",
            thought=f"thought_{id_suffix}",
            response=f"response_{id_suffix}",
            score=score,
            iteration=0
        )


class TestDualMemorySystem(unittest.TestCase):
    """Test integrated dual memory system"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_dual.db"
        self.memory_system = DualMemorySystem(
            episodic_capacity=100,
            shared_capacity=5,
            db_path=str(self.db_path),
            embedding_dim=64
        )
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_problem_lifecycle(self):
        """Test complete problem lifecycle"""
        # Start problem
        problem_id = "test_problem_1"
        shared_mem = self.memory_system.start_problem(problem_id)
        
        self.assertIsNotNone(shared_mem)
        self.assertIn(problem_id, self.memory_system.active_problems)
        
        # Add results to shared memory
        for i in range(3):
            result = IntermediateResult(
                id=f"result_{i}",
                timestamp=datetime.now(),
                domain="test",
                agent_role=f"Agent{i}",
                expression=f"expr_{i}",
                thought="thinking...",
                response="found something",
                score=0.5 + i * 0.1
            )
            shared_mem.add(result)
        
        # End problem and save to episodic
        discovery = self.memory_system.end_problem(problem_id, "test", save_to_episodic=True)
        
        self.assertIsNotNone(discovery)
        self.assertNotIn(problem_id, self.memory_system.active_problems)
        
        # Check discovery was saved
        retrieved = self.memory_system.episodic.get(discovery.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.expression, "expr_2")  # Highest scoring
    
    def test_memory_stats(self):
        """Test memory statistics"""
        # Add some episodic memories
        for i in range(5):
            discovery = Discovery(
                id=f"discovery_{i}",
                timestamp=datetime.now(),
                domain="test",
                expression=f"expr_{i}",
                hypothesis="test",
                evidence=[],
                confidence=0.8,
                validation_score=0.85,
                reasoning_trace=[],
                agent_roles=["Agent1"]
            )
            self.memory_system.episodic.add(discovery)
        
        # Start some problems
        for i in range(3):
            self.memory_system.start_problem(f"problem_{i}")
        
        # Get stats
        stats = self.memory_system.get_memory_stats()
        
        self.assertEqual(stats['episodic']['total_discoveries'], 5)
        self.assertEqual(stats['active_problems'], 3)


class TestMemoryIntegration(unittest.TestCase):
    """Test memory integration features"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.memory_system = DualMemorySystem(
            db_path=str(Path(self.temp_dir) / "test.db")
        )
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_replay_buffer(self):
        """Test memory replay buffer"""
        # Add some discoveries
        for i in range(10):
            discovery = Discovery(
                id=f"disc_{i}",
                timestamp=datetime.now(),
                domain="test",
                expression=f"expr_{i}",
                hypothesis="test",
                evidence=[],
                confidence=0.5 + i * 0.05,
                validation_score=0.6 + i * 0.04,
                reasoning_trace=[],
                agent_roles=["Agent1"]
            )
            self.memory_system.episodic.add(discovery)
        
        # Create replay buffer
        replay_buffer = MemoryReplayBuffer(self.memory_system)
        
        # Sample with prioritization
        samples = replay_buffer.sample(5, min_score=0.7)
        
        self.assertLessEqual(len(samples), 5)
        
        # Check all samples meet minimum score
        for sample in samples:
            self.assertGreaterEqual(sample.validation_score, 0.7)


class TestAdvancedFeatures(unittest.TestCase):
    """Test advanced memory features"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.memory_system = DualMemorySystem(
            db_path=str(Path(self.temp_dir) / "test.db")
        )
        
        # Add test data
        for i in range(20):
            discovery = Discovery(
                id=f"disc_{i}",
                timestamp=datetime.now() - timedelta(days=i*2),
                domain=["attention", "physics"][i % 2],
                expression=f"expr_{i}",
                hypothesis="test",
                evidence=[],
                confidence=np.random.random(),
                validation_score=np.random.random(),
                reasoning_trace=[],
                agent_roles=["Agent1", "Agent2"]
            )
            self.memory_system.episodic.add(discovery)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_consolidation(self):
        """Test memory consolidation"""
        consolidator = MemoryConsolidator(self.memory_system)
        
        # Get initial count
        initial_count = len(self.memory_system.episodic.memories)
        
        # Consolidate old memories (older than 10 days)
        result = consolidator.consolidate_old_memories(age_threshold_days=10)
        
        # Check some memories were consolidated
        self.assertGreater(result['removed_count'], 0)
        
        # Check total count decreased
        final_count = len(self.memory_system.episodic.memories)
        self.assertLess(final_count, initial_count)
    
    def test_export_import(self):
        """Test export and import functionality"""
        exporter = MemoryExporter(self.memory_system)
        
        # Export discoveries
        export_path = Path(self.temp_dir) / "export.zip"
        export_result = exporter.export_discoveries(str(export_path))
        
        self.assertGreater(export_result['exported_count'], 0)
        self.assertTrue(export_path.exists())
        
        # Create new memory system
        new_memory_system = DualMemorySystem(
            db_path=str(Path(self.temp_dir) / "new.db")
        )
        new_exporter = MemoryExporter(new_memory_system)
        
        # Import discoveries
        import_result = new_exporter.import_discoveries(str(export_path))
        
        self.assertEqual(import_result['imported_count'], export_result['exported_count'])
    
    def test_importance_sampling(self):
        """Test importance-based sampling"""
        sampler = ImportanceSampler(self.memory_system)
        
        # Sample with different temperatures
        samples_high_temp = sampler.sample_by_importance(5, temperature=2.0)
        samples_low_temp = sampler.sample_by_importance(5, temperature=0.1)
        
        self.assertEqual(len(samples_high_temp), 5)
        self.assertEqual(len(samples_low_temp), 5)
        
        # Low temperature should give more consistent results
        # (would need multiple runs to test properly)


# Test runner
if __name__ == "__main__":
    unittest.main(verbosity=2)
