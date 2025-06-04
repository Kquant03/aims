#!/usr/bin/env python3
# scripts/benchmark.py - Performance benchmarking tool
import asyncio
import time
import torch
import psutil
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import statistics

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.consciousness import ConsciousnessCore
from src.core.memory_manager import PersistentMemoryManager
from src.utils.gpu_optimizer import gpu_optimizer
from src.utils.logger import setup_logging

logger = setup_logging('INFO', 'logs/benchmark.log')

class AIMSBenchmark:
    """Comprehensive benchmarking for AIMS components"""
    
    def __init__(self):
        self.results = {}
        self.config = {
            'cycle_frequency': 2.0,
            'working_memory_size': 7,
            'coherence_threshold': 0.7
        }
    
    async def benchmark_consciousness_cycle(self, iterations: int = 100):
        """Benchmark consciousness processing cycle"""
        logger.info(f"Benchmarking consciousness cycle ({iterations} iterations)...")
        
        consciousness = ConsciousnessCore(self.config)
        times = []
        
        for i in range(iterations):
            start = time.perf_counter()
            
            # Process input
            consciousness.process_input(f"Test input {i}", {'benchmark': True})
            
            # Run one cycle
            await consciousness._update_attention()
            await consciousness._consolidate_memory()
            await consciousness._update_emotional_state()
            consciousness._calculate_coherence()
            
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        self.results['consciousness_cycle'] = {
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'stdev': statistics.stdev(times) if len(times) > 1 else 0,
            'min': min(times),
            'max': max(times),
            'p95': sorted(times)[int(len(times) * 0.95)],
            'p99': sorted(times)[int(len(times) * 0.99)]
        }
        
        logger.info(f"Consciousness cycle: {self.results['consciousness_cycle']['mean']*1000:.2f}ms average")
    
    async def benchmark_memory_operations(self, num_memories: int = 1000):
        """Benchmark memory storage and retrieval"""
        logger.info(f"Benchmarking memory operations ({num_memories} memories)...")
        
        memory_manager = PersistentMemoryManager(self.config)
        
        # Benchmark storage
        store_times = []
        memory_ids = []
        
        for i in range(num_memories):
            start = time.perf_counter()
            
            memory_id = await memory_manager.store_memory(
                content=f"Test memory content {i} with some additional text to make it realistic",
                context={
                    'user_id': 'benchmark_user',
                    'importance': 0.5 + (i % 5) * 0.1,
                    'emotional_state': {
                        'pleasure': 0.5,
                        'arousal': 0.5,
                        'dominance': 0.5
                    }
                }
            )
            
            elapsed = time.perf_counter() - start
            store_times.append(elapsed)
            memory_ids.append(memory_id)
        
        # Benchmark retrieval
        retrieve_times = []
        
        for i in range(100):  # Sample 100 retrievals
            start = time.perf_counter()
            
            memories = await memory_manager.retrieve_memories(
                f"Test memory content {i % num_memories}",
                k=5
            )
            
            elapsed = time.perf_counter() - start
            retrieve_times.append(elapsed)
        
        self.results['memory_operations'] = {
            'storage': {
                'mean': statistics.mean(store_times),
                'median': statistics.median(store_times),
                'p95': sorted(store_times)[int(len(store_times) * 0.95)]
            },
            'retrieval': {
                'mean': statistics.mean(retrieve_times),
                'median': statistics.median(retrieve_times),
                'p95': sorted(retrieve_times)[int(len(retrieve_times) * 0.95)]
            }
        }
        
        logger.info(f"Memory storage: {self.results['memory_operations']['storage']['mean']*1000:.2f}ms average")
        logger.info(f"Memory retrieval: {self.results['memory_operations']['retrieval']['mean']*1000:.2f}ms average")
    
    def benchmark_gpu_operations(self):
        """Benchmark GPU operations if available"""
        if not torch.cuda.is_available():
            logger.warning("GPU not available, skipping GPU benchmarks")
            return
        
        logger.info("Benchmarking GPU operations...")
        
        # Get GPU info
        gpu_info = gpu_optimizer.get_memory_usage()
        
        # Benchmark matrix operations
        sizes = [512, 1024, 2048, 4096]
        results = {}
        
        for size in sizes:
            # Create random matrices
            a = torch.randn(size, size, device='cuda', dtype=torch.float16)
            b = torch.randn(size, size, device='cuda', dtype=torch.float16)
            
            # Warmup
            for _ in range(10):
                _ = torch.matmul(a, b)
            
            torch.cuda.synchronize()
            
            # Benchmark
            start = time.perf_counter()
            iterations = 100
            
            for _ in range(iterations):
                _ = torch.matmul(a, b)
            
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            results[f'matmul_{size}x{size}'] = elapsed / iterations
            
            # Clean up
            del a, b
            torch.cuda.empty_cache()
        
        self.results['gpu_operations'] = {
            'gpu_info': gpu_info,
            'matrix_operations': results
        }
        
        logger.info(f"GPU Memory: {gpu_info['allocated']:.1f}GB allocated, {gpu_info['free']:.1f}GB free")
    
    def benchmark_system_resources(self):
        """Benchmark system resource usage"""
        logger.info("Benchmarking system resources...")
        
        # CPU info
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        
        # Memory info
        memory = psutil.virtual_memory()
        
        # Disk info
        disk = psutil.disk_usage('/')
        
        self.results['system_resources'] = {
            'cpu': {
                'count': psutil.cpu_count(),
                'usage_percent': statistics.mean(cpu_percent),
                'per_core': cpu_percent
            },
            'memory': {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'percent': memory.percent
            },
            'disk': {
                'total_gb': disk.total / (1024**3),
                'free_gb': disk.free / (1024**3),
                'percent': disk.percent
            }
        }
        
        logger.info(f"CPU Usage: {self.results['system_resources']['cpu']['usage_percent']:.1f}%")
        logger.info(f"Memory: {memory.percent:.1f}% used")
    
    async def run_all_benchmarks(self):
        """Run all benchmarks"""
        logger.info("Starting AIMS benchmark suite...")
        
        # System resources
        self.benchmark_system_resources()
        
        # GPU operations
        self.benchmark_gpu_operations()
        
        # Consciousness processing
        await self.benchmark_consciousness_cycle()
        
        # Memory operations
        await self.benchmark_memory_operations()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print benchmark summary"""
        print("\n" + "="*60)
        print("AIMS BENCHMARK RESULTS")
        print("="*60)
        
        # System info
        sys_info = self.results.get('system_resources', {})
        print(f"\nSystem Resources:")
        print(f"  CPU: {sys_info.get('cpu', {}).get('count', 'N/A')} cores, "
              f"{sys_info.get('cpu', {}).get('usage_percent', 0):.1f}% usage")
        print(f"  Memory: {sys_info.get('memory', {}).get('available_gb', 0):.1f}GB available "
              f"({sys_info.get('memory', {}).get('percent', 0):.1f}% used)")
        
        # GPU info
        if 'gpu_operations' in self.results:
            gpu_info = self.results['gpu_operations']['gpu_info']
            print(f"\nGPU Resources:")
            print(f"  Memory: {gpu_info['allocated']:.1f}GB allocated, "
                  f"{gpu_info['free']:.1f}GB free of {gpu_info['total']:.1f}GB total")
        
        # Consciousness performance
        if 'consciousness_cycle' in self.results:
            cc = self.results['consciousness_cycle']
            print(f"\nConsciousness Cycle Performance:")
            print(f"  Average: {cc['mean']*1000:.2f}ms")
            print(f"  P95: {cc['p95']*1000:.2f}ms")
            print(f"  P99: {cc['p99']*1000:.2f}ms")
            
            # Calculate frequency
            actual_freq = 1.0 / cc['mean']
            print(f"  Max frequency: {actual_freq:.1f}Hz " +
                  f"(target: {self.config['cycle_frequency']}Hz)")
        
        # Memory performance
        if 'memory_operations' in self.results:
            mem = self.results['memory_operations']
            print(f"\nMemory Operations:")
            print(f"  Storage: {mem['storage']['mean']*1000:.2f}ms average, "
                  f"{mem['storage']['p95']*1000:.2f}ms P95")
            print(f"  Retrieval: {mem['retrieval']['mean']*1000:.2f}ms average, "
                  f"{mem['retrieval']['p95']*1000:.2f}ms P95")
        
        print("\n" + "="*60)

async def main():
    """Main benchmark function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AIMS Performance Benchmark')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of iterations for cycle benchmark')
    parser.add_argument('--memories', type=int, default=1000,
                       help='Number of memories for memory benchmark')
    parser.add_argument('--gpu', action='store_true',
                       help='Run GPU benchmarks')
    
    args = parser.parse_args()
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    benchmark = AIMSBenchmark()
    
    if args.gpu:
        benchmark.benchmark_gpu_operations()
    else:
        await benchmark.run_all_benchmarks()

if __name__ == "__main__":
    asyncio.run(main())