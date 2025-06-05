# gpu_optimizer.py - Fixed version
import torch
import logging
from typing import Dict, Any, Optional
import psutil
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class GPUOptimizer:
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
        
        # Get GPU info using torch instead of pynvml
        self.device = torch.device('cuda:0')
        torch.cuda.set_device(0)
        
        # Get VRAM info
        gpu_properties = torch.cuda.get_device_properties(0)
        self.vram_total = gpu_properties.total_memory
        
        logger.info(f"GPU initialized: {gpu_properties.name}")
        logger.info(f"Total VRAM: {self.vram_total / 1e9:.1f} GB")
        
        # Enable TF32 for Ampere GPUs (RTX 3090)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(0.9)
        torch.cuda.empty_cache()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage"""
        if not torch.cuda.is_available():
            return {'allocated': 0, 'reserved': 0, 'free': 0}
        
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        free = (self.vram_total - torch.cuda.memory_reserved()) / 1e9
        
        return {
            'allocated': allocated,
            'reserved': reserved,
            'free': free
        }
    
    def optimize_model_for_inference(self, model: torch.nn.Module, 
                                   use_half_precision: bool = True) -> torch.nn.Module:
        """Optimize model for inference"""
        if not torch.cuda.is_available():
            return model
        
        model = model.to(self.device)
        model.eval()
        
        if use_half_precision:
            model = model.half()
        
        # Enable cudnn benchmarking
        torch.backends.cudnn.benchmark = True
        
        return model
    
    def get_optimal_batch_size(self, model: torch.nn.Module, 
                             input_shape: tuple, 
                             target_memory_usage: float = 0.8) -> int:
        """Calculate optimal batch size for given model and input"""
        if not torch.cuda.is_available():
            return 1
        
        # Start with batch size 1 and increase
        batch_size = 1
        
        while True:
            try:
                # Clear cache
                torch.cuda.empty_cache()
                
                # Try forward pass
                test_input = torch.randn(batch_size, *input_shape).to(self.device)
                if hasattr(model, 'half'):
                    test_input = test_input.half()
                
                with torch.no_grad():
                    _ = model(test_input)
                
                # Check memory usage
                memory_usage = self.get_memory_usage()
                used_fraction = memory_usage['reserved'] / (self.vram_total / 1e9)
                
                if used_fraction > target_memory_usage:
                    # We've exceeded target, use previous batch size
                    return max(1, batch_size - 1)
                
                # Try larger batch
                batch_size *= 2
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # OOM, use previous batch size
                    return max(1, batch_size // 2)
                else:
                    raise
    
    @contextmanager
    def mixed_precision_context(self):
        """Context manager for mixed precision operations"""
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                yield
        else:
            yield

gpu_optimizer = GPUOptimizer()