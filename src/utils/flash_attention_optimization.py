# flash_attention_optimization.py - Fixed version
"""
GPU-Optimized Flash Attention (with fallbacks for missing libraries)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import math
import logging
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

# Try to import Flash Attention
try:
    from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
    FLASH_ATTENTION_AVAILABLE = True
    logger.info("Flash Attention 2 is available")
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    logger.warning("Flash Attention not available, using PyTorch native attention")

# Check for PyTorch 2.0+ scaled_dot_product_attention
PYTORCH_2_AVAILABLE = hasattr(F, 'scaled_dot_product_attention')

@dataclass
class AttentionConfig:
    """Configuration for optimized attention"""
    hidden_size: int = 768
    num_attention_heads: int = 12
    attention_dropout: float = 0.1
    use_flash_attention: bool = True
    use_bf16: bool = True
    max_sequence_length: int = 16384
    window_size: Optional[int] = None
    use_rotary_embeddings: bool = True
    use_alibi: bool = False

class OptimizedMultiHeadAttention(nn.Module):
    """Multi-head attention with optimizations"""
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scaling = self.head_dim ** -0.5
        
        assert self.hidden_size % self.num_heads == 0
        
        # Linear projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        
        if config.use_rotary_embeddings:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=config.max_sequence_length
            )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass"""
        batch_size, seq_length, _ = hidden_states.shape
        
        # Linear projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = self._reshape_qkv(query_states, batch_size, seq_length)
        key_states = self._reshape_qkv(key_states, batch_size, seq_length)
        value_states = self._reshape_qkv(value_states, batch_size, seq_length)
        
        # Apply rotary embeddings if enabled
        if self.config.use_rotary_embeddings and position_ids is not None:
            cos, sin = self.rotary_emb(value_states, seq_len=seq_length)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )
        
        # Handle past key values
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
        
        present_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if use_cache:
            present_key_value = (key_states, value_states)
        
        # Choose attention implementation
        if FLASH_ATTENTION_AVAILABLE and self.config.use_flash_attention and not output_attentions:
            attn_output = self._flash_attention_forward(
                query_states, key_states, value_states, attention_mask
            )
            attn_weights = None
        else:
            attn_output, attn_weights = self._standard_attention_forward(
                query_states, key_states, value_states, attention_mask, output_attentions
            )
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, self.hidden_size)
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights, present_key_value
    
    def _reshape_qkv(self, tensor: torch.Tensor, batch_size: int, seq_length: int) -> torch.Tensor:
        """Reshape tensor for multi-head attention"""
        return tensor.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
    
    def _flash_attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Flash Attention forward pass"""
        # Placeholder - implement based on available version
        return self._standard_attention_forward(
            query_states, key_states, value_states, attention_mask, False
        )[0]
    
    def _standard_attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Standard attention implementation"""
        # Calculate attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1))
        attn_weights = attn_weights * self.scaling
        
        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.attention_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        
        return attn_output, attn_weights if output_attentions else None

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding"""
    
    def __init__(self, dim: int, max_position_embeddings: int = 16384, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(max_position_embeddings)
    
    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None):
        if seq_len is not None and seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        
        seq_len = seq_len or x.shape[1]
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype)
        )

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, 
                        sin: torch.Tensor, position_ids: torch.Tensor):
    """Apply rotary position embeddings"""
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

class AttentionBenchmark:
    """Benchmark different attention implementations"""
    
    @staticmethod
    def benchmark_attention(
        batch_size: int = 4,
        seq_lengths: List[int] = None,
        hidden_size: int = 768,
        num_heads: int = 12
    ) -> Dict[str, Any]:
        """Benchmark attention performance"""
        if seq_lengths is None:
            seq_lengths = [512, 1024, 2048]
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        results = {}
        
        for seq_len in seq_lengths:
            input_tensor = torch.randn(
                batch_size, seq_len, hidden_size, 
                device=device, dtype=torch.float32
            )
            
            config = AttentionConfig(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                use_flash_attention=False
            )
            
            model = OptimizedMultiHeadAttention(config).to(device)
            model.eval()
            
            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    _ = model(input_tensor)
            
            # Benchmark
            start = time.time()
            with torch.no_grad():
                for _ in range(10):
                    _ = model(input_tensor)
            
            elapsed = (time.time() - start) / 10
            
            results[f'seq{seq_len}'] = {
                'time_ms': elapsed * 1000,
                'device': str(device)
            }
        
        return results
