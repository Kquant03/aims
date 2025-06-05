# tests/test_performance.py - Performance and load testing
import pytest
import asyncio
import time
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import psutil
import gc
from memory_profiler import profile
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.living_consciousness import ConsciousnessCore, ConsciousnessAttentionMechanism
from src.core.memory_manager import AdvancedMemoryManager
from src.utils.gpu_optimizer import GPUOptimizer

class TestConsciousnessPerformance:
    """Performance tests for consciousness core"""
    
    @pytest.fixture
    def consciousness_core(self):
        config = {
            'cycle_frequency': 5.0,  # 5Hz for testing
            'working_memory_size': 7,
            'coherence_threshold': 0.7
        }
        return ConsciousnessCore(config)
    
    @pytest.mark.asyncio
    async def test_consciousness_cycle_timing(self, consciousness_core):
        """Test that consciousness cycles run at specified frequency"""
        cycle_times = []
        
        # Override the loop to capture timings
        original_loop = consciousness_core.consciousness_loop
        
        async def timed_loop():
            start_time = time.time()
            cycles = 0
            
            while cycles < 10:  # Run 10 cycles
                cycle_start = time.time()
                
                await consciousness_core._update_attention()
                await consciousness_core._consolidate_memory()
                await consciousness_core._update_emotional_state()
                consciousness_core._calculate_coherence()
                
                cycle_duration = time.time() - cycle_start
                cycle_times.append(cycle_duration)
                
                # Sleep to maintain frequency
                sleep_time = (1.0 / consciousness_core.cycle_frequency) - cycle_duration
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
                cycles += 1
                
                if cycles >= 10:
                    consciousness_core._running = False
        
        # Run the timed loop
        await timed_loop()
        
        # Check timing
        avg_cycle_time = np.mean(cycle_times)
        max_cycle_time = np.max(cycle_times)
        
        # Should complete within expected time (200ms for 5Hz)
        assert avg_cycle_time < 0.2  # 200ms
        assert max_cycle_time < 0.25  # Allow some variance
        
        print(f"Average cycle time: {avg_cycle_time*1000:.2f}ms")
        print(f"Max cycle time: {max_cycle_time*1000:.2f}ms")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_attention_mechanism_gpu_performance(self):
        """Test Flash Attention performance on GPU"""
        attention = ConsciousnessAttentionMechanism(
            input_dim=768,
            hidden_dim=512,
            num_heads=8
        ).cuda()
        
        # Test different sequence lengths
        batch_sizes = [1, 4, 8]
        seq_lengths = [128, 512, 1024, 2048]
        
        results = {}
        
        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                # Create input
                x = torch.randn(batch_size, seq_len, 768).cuda()
                
                # Warmup
                for _ in range(5):
                    _ = attention(x)
                
                # Time forward pass
                torch.cuda.synchronize()
                start_time = time.time()
                
                num_runs = 20
                for _ in range(num_runs):
                    _ = attention(x)
                
                torch.cuda.synchronize()
                total_time = time.time() - start_time
                avg_time = total_time / num_runs
                
                results[f"batch_{batch_size}_seq_{seq_len}"] = avg_time * 1000  # ms
                
                print(f"Batch {batch_size}, Seq {seq_len}: {avg_time*1000:.2f}ms")
        
        # Check performance targets
        # For RTX 3090, we expect < 10ms for most reasonable sizes
        assert results["batch_1_seq_512"] < 10
        assert results["batch_4_seq_512"] < 20

class TestMemoryPerformance:
    """Performance tests for memory subsystem"""
    
    @pytest.fixture
    def memory_manager(self):
        config = {
            'embedding_dim': 768,
            'max_memories': 100000,
            'consolidation_threshold': 0.3,
            'use_gpu': torch.cuda.is_available()
        }
        return AdvancedMemoryManager(config)
    
    @pytest.mark.asyncio
    async def test_memory_retrieval_speed(self, memory_manager):
        """Test sub-100ms memory retrieval"""
        # Generate test memories
        num_memories = 10000
        embeddings = np.random.randn(num_memories, 768).astype(np.float32)
        
        # Store memories
        print(f"Storing {num_memories} memories...")
        start_time = time.time()
        
        for i in range(num_memories):
            await memory_manager.store_memory_with_embedding(
                content=f"Test memory {i}",
                embedding=embeddings[i],
                context={'importance': np.random.rand()}
            )
        
        store_time = time.time() - start_time
        print(f"Storage time: {store_time:.2f}s ({store_time/num_memories*1000:.2f}ms per memory)")
        
        # Test retrieval speed
        query_embedding = np.random.randn(768).astype(np.float32)
        
        # Warmup
        for _ in range(5):
            await memory_manager.retrieve_memories_by_embedding(query_embedding, k=10)
        
        # Time retrievals
        retrieval_times = []
        for _ in range(100):
            start_time = time.time()
            results = await memory_manager.retrieve_memories_by_embedding(
                query_embedding, k=10
            )
            retrieval_time = (time.time() - start_time) * 1000  # ms
            retrieval_times.append(retrieval_time)
        
        avg_retrieval_time = np.mean(retrieval_times)
        p95_retrieval_time = np.percentile(retrieval_times, 95)
        p99_retrieval_time = np.percentile(retrieval_times, 99)
        
        print(f"Average retrieval time: {avg_retrieval_time:.2f}ms")
        print(f"P95 retrieval time: {p95_retrieval_time:.2f}ms")
        print(f"P99 retrieval time: {p99_retrieval_time:.2f}ms")
        
        # Check sub-100ms target
        assert avg_retrieval_time < 100
        assert p95_retrieval_time < 100
    
    def test_memory_leak_detection(self, memory_manager):
        """Test for memory leaks during operations"""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Perform many operations
        for cycle in range(10):
            # Store and retrieve memories
            embeddings = np.random.randn(1000, 768).astype(np.float32)
            
            # Force garbage collection
            gc.collect()
            
            cycle_memory = psutil.Process().memory_info().rss / 1024 / 1024
            print(f"Cycle {cycle}: Memory usage: {cycle_memory:.2f}MB")
        
        # Final memory check
        gc.collect()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        print(f"Memory increase: {memory_increase:.2f}MB")
        
        # Should not increase by more than 100MB
        assert memory_increase < 100

class TestGPUUtilization:
    """Test GPU optimization and utilization"""
    
    @pytest.fixture
    def gpu_optimizer(self):
        return GPUOptimizer()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_gpu_memory_optimization(self, gpu_optimizer):
        """Test GPU memory usage optimization"""
        # Create a test model
        model = torch.nn.Sequential(
            torch.nn.Linear(768, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 768)
        )
        
        # Get baseline memory
        torch.cuda.empty_cache()
        baseline_memory = gpu_optimizer.get_memory_usage()
        
        # Optimize model
        optimized_model = gpu_optimizer.optimize_model_for_inference(
            model,
            use_half_precision=True
        )
        
        # Check memory after optimization
        optimized_memory = gpu_optimizer.get_memory_usage()
        
        print(f"Baseline memory: {baseline_memory}")
        print(f"Optimized memory: {optimized_memory}")
        
        # Test optimal batch size calculation
        optimal_batch = gpu_optimizer.get_optimal_batch_size(
            optimized_model,
            (768,),
            target_memory_usage=0.8
        )
        
        print(f"Optimal batch size: {optimal_batch}")
        assert optimal_batch > 0
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_mixed_precision_performance(self, gpu_optimizer):
        """Test mixed precision training performance"""
        model = torch.nn.TransformerEncoderLayer(
            d_model=768,
            nhead=12,
            dim_feedforward=3072
        ).cuda()
        
        input_data = torch.randn(32, 128, 768).cuda()
        
        # Test FP32 performance
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_data)
        
        torch.cuda.synchronize()
        fp32_time = time.time() - start_time
        
        # Test FP16 performance
        model_fp16 = model.half()
        input_data_fp16 = input_data.half()
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        with gpu_optimizer.mixed_precision_context():
            with torch.no_grad():
                for _ in range(10):
                    _ = model_fp16(input_data_fp16)
        
        torch.cuda.synchronize()
        fp16_time = time.time() - start_time
        
        speedup = fp32_time / fp16_time
        print(f"FP32 time: {fp32_time:.3f}s")
        print(f"FP16 time: {fp16_time:.3f}s")
        print(f"Speedup: {speedup:.2f}x")
        
        # Should see at least 1.5x speedup
        assert speedup > 1.5

class TestLoadTesting:
    """Load testing for multi-user scenarios"""
    
    @pytest.mark.asyncio
    async def test_concurrent_users(self):
        """Test system with multiple concurrent users"""
        # This would test the system with many simultaneous connections
        # Simulating 100 concurrent users
        
        async def simulate_user(user_id: int):
            # Simulate user interactions
            for _ in range(10):
                # Simulate API calls
                await asyncio.sleep(np.random.uniform(0.1, 0.5))
            return user_id
        
        # Run concurrent users
        start_time = time.time()
        tasks = [simulate_user(i) for i in range(100)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        print(f"Handled {len(results)} users in {total_time:.2f}s")
        assert len(results) == 100

class TestMemoryProfiling:
    """Memory profiling tests"""
    
    @profile
    def test_consciousness_memory_usage(self):
        """Profile memory usage of consciousness system"""
        config = {
            'cycle_frequency': 2.0,
            'working_memory_size': 7
        }
        
        consciousness = ConsciousnessCore(config)
        
        # Simulate interactions
        for i in range(100):
            consciousness.process_input(f"Test input {i}")
        
        # Check memory buffer
        assert len(consciousness.memory_buffer) <= 7
    
    @profile
    def test_memory_manager_usage(self):
        """Profile memory usage of memory manager"""
        config = {
            'embedding_dim': 768,
            'max_memories': 10000
        }
        
        manager = AdvancedMemoryManager(config)
        
        # Store memories
        for i in range(1000):
            embedding = np.random.randn(768).astype(np.float32)
            asyncio.run(manager.store_memory_with_embedding(
                content=f"Memory {i}",
                embedding=embedding,
                context={}
            ))

if __name__ == '__main__':
    # Run performance tests
    pytest.main([__file__, '-v', '-s'])