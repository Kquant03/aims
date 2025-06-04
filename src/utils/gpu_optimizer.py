# gpu_optimizer.py - GPU Memory and Performance Optimization for RTX 3090
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
import logging
import psutil
import pynvml
from contextlib import contextmanager
import gc
import functools

logger = logging.getLogger(__name__)

class GPUOptimizer:
    """Optimizes GPU usage for RTX 3090 24GB VRAM"""
    def verify_flash_attention(self) -> Dict[str, Any]:
        """Comprehensive Flash Attention verification"""
        benchmark = FlashAttentionV2Benchmark()
        
        results = {
            'available': benchmark.flash_available,
            'numerical_accuracy': False,
            'benchmarks': {}
        }
        
        if benchmark.flash_available:
            # Verify accuracy
            results['numerical_accuracy'] = benchmark.verify_numerical_accuracy()
            
            # Run benchmarks for different sizes
            test_configs = [
                (8, 512, 768, 12),    # Small
                (4, 2048, 768, 12),   # Medium
                (2, 4096, 768, 12),   # Large
            ]
            
            for batch, seq_len, hidden, heads in test_configs:
                key = f"batch{batch}_seq{seq_len}"
                results['benchmarks'][key] = benchmark.benchmark_attention(
                    batch, seq_len, hidden, heads
                )
        
        return results
    
    def __init__(self):
        self.device = None
        self.vram_total = 0
        self.initialize_gpu()
        
    def initialize_gpu(self):
        """Initialize GPU monitoring and settings"""
        if not torch.cuda.is_available():
            logger.warning("No GPU detected, running on CPU")
            self.device = torch.device('cpu')
            return
        
        # Initialize NVML for GPU monitoring
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.vram_total = info.total
            
            logger.info(f"GPU initialized: {pynvml.nvmlDeviceGetName(handle).decode()}")
            logger.info(f"Total VRAM: {self.vram_total / 1e9:.1f} GB")
        except Exception as e:
            logger.error(f"Error initializing NVML: {e}")
        
        # Set CUDA device
        self.device = torch.device('cuda:0')
        torch.cuda.set_device(0)
        
        # Enable TF32 for Ampere GPUs (RTX 3090)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cudnn autotuner
        torch.backends.cudnn.benchmark = True
        
        # Set memory fraction to prevent OOM
        torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of VRAM
        
        # Clear any existing allocations
        torch.cuda.empty_cache()
        gc.collect()

        
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage"""
        if not torch.cuda.is_available():
            return {'allocated': 0, 'reserved': 0, 'free': 0}
        
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        
        # Get free memory from NVML
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free = info.free / 1e9
        except:
            free = (self.vram_total / 1e9) - reserved
        
        return {
            'allocated': allocated,
            'reserved': reserved,
            'free': free,
            'total': self.vram_total / 1e9
        }
    
    @contextmanager
    def memory_efficient_mode(self):
        """Context manager for memory-efficient operations"""
        # Store original settings
        original_grad_enabled = torch.is_grad_enabled()
        
        try:
            # Disable gradients
            torch.set_grad_enabled(False)
            
            # Clear cache
            torch.cuda.empty_cache()
            
            yield
            
        finally:
            # Restore settings
            torch.set_grad_enabled(original_grad_enabled)
            
            # Clean up
            torch.cuda.empty_cache()
            gc.collect()
    
    def optimize_model_for_inference(self, model: nn.Module, 
                                   use_half_precision: bool = True,
                                   use_torch_compile: bool = True) -> nn.Module:
        """Optimize a model for inference on RTX 3090"""
        model.eval()
        
        # Move to GPU
        model = model.to(self.device)
        
        # Convert to half precision for faster inference
        if use_half_precision and self.device.type == 'cuda':
            model = model.half()
            logger.info("Model converted to FP16")
        
        # Use torch.compile for additional optimizations (PyTorch 2.0+)
        if use_torch_compile and hasattr(torch, 'compile'):
            model = torch.compile(model, mode='reduce-overhead')
            logger.info("Model compiled with torch.compile")
        
        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    def apply_gradient_checkpointing(self, model: nn.Module) -> nn.Module:
        """Apply gradient checkpointing to reduce memory usage during training"""
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        return model
    
    def get_optimal_batch_size(self, model: nn.Module, 
                              input_shape: Tuple[int, ...],
                              target_memory_usage: float = 0.8) -> int:
        """Find optimal batch size for given model and input"""
        if self.device.type == 'cpu':
            return 4  # Conservative default for CPU
        
        # Start with batch size 1
        batch_size = 1
        test_input = torch.randn(batch_size, *input_shape).to(self.device)
        
        # Get base memory usage
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        base_memory = torch.cuda.memory_allocated()
        
        try:
            # Forward pass with batch size 1
            with torch.no_grad():
                _ = model(test_input)
            
            torch.cuda.synchronize()
            single_batch_memory = torch.cuda.memory_allocated() - base_memory
            
            # Calculate maximum batch size
            memory_info = self.get_memory_usage()
            available_memory = memory_info['free'] * 1e9 * target_memory_usage
            
            max_batch_size = int(available_memory / single_batch_memory)
            
            # Verify with actual test
            test_batch_size = min(max_batch_size, 128)  # Cap at 128 for testing
            test_input = torch.randn(test_batch_size, *input_shape).to(self.device)
            
            with torch.no_grad():
                _ = model(test_input)
            
            logger.info(f"Optimal batch size: {max_batch_size}")
            return max_batch_size
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning("OOM during batch size calculation, using conservative estimate")
                return 1
            raise
        finally:
            torch.cuda.empty_cache()
    
    def profile_memory_usage(self, func):
        """Decorator to profile GPU memory usage of a function"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.device.type == 'cpu':
                return func(*args, **kwargs)
            
            # Clear cache and get initial memory
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            start_memory = torch.cuda.memory_allocated()
            
            # Run function
            result = func(*args, **kwargs)
            
            # Get final memory
            torch.cuda.synchronize()
            end_memory = torch.cuda.memory_allocated()
            
            memory_used = (end_memory - start_memory) / 1e6  # MB
            logger.info(f"{func.__name__} used {memory_used:.2f} MB of GPU memory")
            
            return result
        
        return wrapper
    
    @contextmanager
    def mixed_precision_context(self, enabled: bool = True):
        """Context manager for automatic mixed precision"""
        if not enabled or self.device.type == 'cpu':
            yield
            return
        
        # Use automatic mixed precision
        with torch.cuda.amp.autocast(enabled=True):
            yield
    
    def optimize_attention_computation(self, 
                                     query: torch.Tensor,
                                     key: torch.Tensor,
                                     value: torch.Tensor,
                                     use_flash_attention: bool = True) -> torch.Tensor:
        """Optimize attention computation for RTX 3090"""
        batch_size, seq_len, hidden_dim = query.shape
        
        # Use Flash Attention if available and beneficial
        if use_flash_attention and seq_len > 1024:
            try:
                # Flash Attention implementation
                # This would use a library like xformers or flash-attn
                logger.debug("Using Flash Attention")
                # Placeholder for actual implementation
                return self._standard_attention(query, key, value)
            except:
                logger.debug("Flash Attention not available, using standard")
                return self._standard_attention(query, key, value)
        else:
            return self._standard_attention(query, key, value)
    
    def _standard_attention(self, query: torch.Tensor,
                          key: torch.Tensor,
                          value: torch.Tensor) -> torch.Tensor:
        """Standard attention computation"""
        scale = query.size(-1) ** -0.5
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, value)
    
    def estimate_model_memory(self, model: nn.Module, 
                            input_shape: Tuple[int, ...],
                            batch_size: int = 1) -> Dict[str, float]:
        """Estimate memory requirements for a model"""
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        param_memory = total_params * 4 / 1e9  # Assuming FP32
        
        # Estimate activation memory (rough approximation)
        # This is highly model-dependent
        activation_memory = batch_size * 0.1  # 100MB per batch (rough estimate)
        
        # Estimate gradient memory (if training)
        gradient_memory = param_memory if any(p.requires_grad for p in model.parameters()) else 0
        
        return {
            'parameters': param_memory,
            'activations': activation_memory,
            'gradients': gradient_memory,
            'total': param_memory + activation_memory + gradient_memory
        }
    
    def cleanup(self):
        """Cleanup GPU resources"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Shutdown NVML
        try:
            pynvml.nvmlShutdown()
        except:
            pass

        

class FlashAttentionV2Benchmark:
    """Benchmarking and verification for Flash Attention v2"""
    
    def __init__(self):
        self.flash_available = self._check_flash_attention()
        
    def _check_flash_attention(self) -> bool:
        """Check if Flash Attention v2 is available and working"""
        try:
            import flash_attn
            from flash_attn import flash_attn_func
            
            # Check version
            version = getattr(flash_attn, '__version__', 'unknown')
            logger.info(f"Flash Attention version: {version}")
            
            # Test basic functionality
            test_tensor = torch.randn(1, 8, 64, device='cuda', dtype=torch.float16)
            _ = flash_attn_func(test_tensor, test_tensor, test_tensor)
            
            logger.info("Flash Attention v2 is available and functional")
            return True
            
        except ImportError:
            logger.warning("Flash Attention not installed")
            return False
        except Exception as e:
            logger.error(f"Flash Attention test failed: {e}")
            return False
    
    def benchmark_attention(self, batch_size: int = 8, seq_len: int = 2048, 
                          hidden_dim: int = 768, num_heads: int = 12) -> Dict[str, float]:
        """Benchmark Flash Attention vs standard attention"""
        import time
        
        results = {}
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Prepare inputs
        head_dim = hidden_dim // num_heads
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                       device=device, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        
        # Warmup
        for _ in range(10):
            if self.flash_available:
                _ = flash_attn_func(q, k, v)
            else:
                scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
                attn = torch.softmax(scores, dim=-1)
                _ = torch.matmul(attn, v)
        
        # Benchmark Flash Attention
        if self.flash_available:
            torch.cuda.synchronize()
            start = time.time()
            
            for _ in range(100):
                output = flash_attn_func(q, k, v)
            
            torch.cuda.synchronize()
            results['flash_attention_ms'] = (time.time() - start) / 100 * 1000
        
        # Benchmark standard attention
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(100):
            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
            attn = torch.softmax(scores, dim=-1)
            output = torch.matmul(attn, v)
        
        torch.cuda.synchronize()
        results['standard_attention_ms'] = (time.time() - start) / 100 * 1000
        
        # Calculate speedup
        if self.flash_available:
            results['speedup'] = results['standard_attention_ms'] / results['flash_attention_ms']
        
        # Memory usage
        results['peak_memory_mb'] = torch.cuda.max_memory_allocated() / 1e6
        
        return results
    
    def verify_numerical_accuracy(self, tolerance: float = 1e-3) -> bool:
        """Verify Flash Attention numerical accuracy"""
        if not self.flash_available:
            return False
        
        # Test inputs
        batch_size, seq_len, num_heads, head_dim = 2, 512, 8, 64
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                       device='cuda', dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        
        # Flash Attention output
        flash_output = flash_attn_func(q, k, v)
        
        # Standard attention output
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        standard_output = torch.matmul(attn, v)
        
        # Compare outputs
        max_diff = torch.max(torch.abs(flash_output - standard_output)).item()
        mean_diff = torch.mean(torch.abs(flash_output - standard_output)).item()
        
        logger.info(f"Flash Attention accuracy - Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
        
        return max_diff < tolerance

class MemoryEfficientAttention(nn.Module):
    """Memory-efficient attention implementation for RTX 3090"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                use_checkpoint: bool = False) -> torch.Tensor:
        """Forward pass with optional gradient checkpointing"""
        if use_checkpoint and x.requires_grad:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, mask
            )
        else:
            return self._forward_impl(x, mask)
    
    def _forward_impl(self, x: torch.Tensor, 
                     mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch, heads, seq, dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Transpose back and reshape
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)
        
        # Final projection
        output = self.out_proj(attn_output)
        
        return output


# Singleton instance
gpu_optimizer = GPUOptimizer()