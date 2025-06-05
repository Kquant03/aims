# gpu_optimizer.py - Fixed version
import torch
import logging
from typing import Dict, Any, Optional
import psutil

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

gpu_optimizer = GPUOptimizer()