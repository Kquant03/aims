# src/core/flash_attention_optimization.py
"""
Phase 8: GPU-Optimized Flash Attention
Optimized attention mechanisms for RTX 3090 with Flash Attention 2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math
import logging
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

# Check for Flash Attention availability
try:
    from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
    from flash_attn.bert_padding import pad_input, unpad_input
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
    use_bf16: bool = True  # BFloat16 for Ampere GPUs
    max_sequence_length: int = 16384  # Supported by Flash Attention 2
    window_size: Optional[int] = None  # For sliding window attention
    use_rotary_embeddings: bool = True
    use_alibi: bool = False  # Alternative positional encoding
    
class OptimizedMultiHeadAttention(nn.Module):
    """
    Multi-head attention optimized for RTX 3090 with Flash Attention 2
    """
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scaling = self.head_dim ** -0.5
        
        # Ensure hidden size is divisible by num_heads
        assert self.hidden_size % self.num_heads == 0
        
        # Linear projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Dropout
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        
        # Rotary embeddings if enabled
        if config.use_rotary_embeddings:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=config.max_sequence_length
            )
        
        # ALiBi slopes if enabled
        if config.use_alibi:
            self.alibi_slopes = self._get_alibi_slopes()
            
        # Check device and optimization availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._validate_optimization_support()
        
    def _validate_optimization_support(self):
        """Validate GPU and optimization support"""
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            compute_capability = torch.cuda.get_device_capability(0)
            
            logger.info(f"GPU: {device_name}, Compute Capability: {compute_capability}")
            
            # RTX 3090 has compute capability 8.6
            if compute_capability[0] >= 8:
                logger.info("Ampere or newer GPU detected - full optimization support")
                if self.config.use_bf16:
                    # Enable TF32 for Ampere
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
            else:
                logger.warning("Older GPU detected - some optimizations may not be available")
                self.config.use_bf16 = False
    
    def _get_alibi_slopes(self) -> torch.Tensor:
        """Calculate ALiBi slopes for each attention head"""
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio ** i for i in range(n)]
            
            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                return (get_slopes_power_of_2(closest_power_of_2) + 
                       get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2])
        
        slopes = torch.tensor(get_slopes(self.num_heads)).to(self.device)
        return slopes
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass with Flash Attention optimization
        """
        batch_size, seq_length, _ = hidden_states.shape
        
        # Use bfloat16 if configured and available
        if self.config.use_bf16 and hidden_states.dtype != torch.bfloat16:
            hidden_states = hidden_states.to(torch.bfloat16)
        
        # Linear projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = self._reshape_qkv(query_states, batch_size, seq_length)
        key_states = self._reshape_qkv(key_states, batch_size, seq_length)
        value_states = self._reshape_qkv(value_states, batch_size, seq_length)
        
        # Apply rotary embeddings if enabled
        if self.config.use_rotary_embeddings:
            cos, sin = self.rotary_emb(value_states, seq_len=seq_length)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )
        
        # Handle past key values for caching
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
        
        if use_cache:
            present_key_value = (key_states, value_states)
        else:
            present_key_value = None
        
        # Choose attention implementation
        if FLASH_ATTENTION_AVAILABLE and self.config.use_flash_attention and not output_attentions:
            attn_output = self._flash_attention_forward(
                query_states, key_states, value_states, attention_mask
            )
        elif PYTORCH_2_AVAILABLE:
            attn_output = self._pytorch2_attention_forward(
                query_states, key_states, value_states, attention_mask
            )
        else:
            attn_output = self._standard_attention_forward(
                query_states, key_states, value_states, attention_mask, output_attentions
            )
            
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, self.hidden_size)
        attn_output = self.out_proj(attn_output)
        
        return attn_output, None, present_key_value
    
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
        """
        Flash Attention 2 forward pass
        Optimized for RTX 3090 with 24GB VRAM
        """
        batch_size = query_states.shape[0]
        
        # Reshape for Flash Attention (batch, seq_len, num_heads, head_dim)
        q = query_states.transpose(1, 2)
        k = key_states.transpose(1, 2)
        v = value_states.transpose(1, 2)
        
        # Handle attention mask and padding
        if attention_mask is not None:
            # Flash Attention uses different mask format
            # Convert boolean mask to attention bias
            if attention_mask.dtype == torch.bool:
                # Assuming mask is (batch, seq_len) where True means attend
                seqlens = attention_mask.sum(dim=1).int()
            else:
                # Assuming standard attention mask
                seqlens = torch.ones(batch_size, dtype=torch.int32) * key_states.shape[2]
        else:
            seqlens = torch.ones(batch_size, dtype=torch.int32) * key_states.shape[2]
        
        # Call Flash Attention
        # Note: Flash Attention 2 supports:
        # - Sequence length up to 256K
        # - Causal and non-causal attention
        # - Sliding window attention
        # - ALiBi positional bias
        
        dropout_p = self.config.attention_dropout if self.training else 0.0
        
        attn_output = flash_attn_func(
            q, k, v,
            dropout_p=dropout_p,
            softmax_scale=self.scaling,
            causal=False,  # Set to True for autoregressive models
            window_size=(self.config.window_size, 0) if self.config.window_size else (-1, -1),
            alibi_slopes=self.alibi_slopes if self.config.use_alibi else None,
        )
        
        return attn_output
    
    def _pytorch2_attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        PyTorch 2.0+ native scaled_dot_product_attention
        Automatically uses Flash Attention when available
        """
        # Apply scaling
        query_states = query_states * self.scaling
        
        # PyTorch 2.0 attention
        dropout_p = self.config.attention_dropout if self.training else 0.0
        
        # Convert attention mask to correct format
        if attention_mask is not None:
            # Expand mask for all heads
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.expand(
                query_states.shape[0], self.num_heads, 
                query_states.shape[2], key_states.shape[2]
            )
        
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=False
        )
        
        return attn_output
    
    def _standard_attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Standard attention implementation (fallback)
        """
        # Calculate attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1))
        attn_weights = attn_weights * self.scaling
        
        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Apply ALiBi if enabled
        if self.config.use_alibi:
            position_bias = self._compute_alibi_bias(attn_weights.shape[-2], attn_weights.shape[-1])
            attn_weights = attn_weights + position_bias
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.attention_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        
        return attn_output, attn_weights if output_attentions else None
    
    def _compute_alibi_bias(self, query_length: int, key_length: int) -> torch.Tensor:
        """Compute ALiBi positional bias"""
        relative_positions = torch.arange(query_length)[:, None] - torch.arange(key_length)[None, :]
        relative_positions = relative_positions.to(self.device)
        
        alibi = self.alibi_slopes[:, None, None] * relative_positions[None, :, :]
        return alibi

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding"""
    
    def __init__(self, dim: int, max_position_embeddings: int = 16384, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Build cache
        self._set_cos_sin_cache(max_position_embeddings)
    
    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
            
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype)
        )

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, 
                        sin: torch.Tensor, position_ids: torch.Tensor):
    """Apply rotary position embeddings to query and key tensors"""
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

class ConsciousnessAttentionMechanism(nn.Module):
    """
    Complete consciousness attention mechanism with Flash Attention optimization
    """
    
    def __init__(self, config: AttentionConfig, num_layers: int = 12):
        super().__init__()
        self.config = config
        self.num_layers = num_layers
        
        # Create attention layers
        self.attention_layers = nn.ModuleList([
            OptimizedMultiHeadAttention(config) for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_size, eps=1e-5) for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            self._create_ffn(config.hidden_size) for _ in range(num_layers)
        ])
        
        # Consciousness-specific attention scoring
        self.consciousness_scorer = nn.Linear(config.hidden_size, 1)
        
    def _create_ffn(self, hidden_size: int) -> nn.Module:
        """Create feed-forward network"""
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(0.1)
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        consciousness_state: Optional[Dict[str, Any]] = None,
        use_cache: bool = False,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through consciousness attention mechanism
        """
        all_hidden_states = []
        all_attentions = []
        consciousness_scores = []
        
        # Track performance metrics
        start_time = time.time()
        
        # Process through layers
        for i, (attn_layer, ln, ffn) in enumerate(
            zip(self.attention_layers, self.layer_norms, self.ffn_layers)
        ):
            # Pre-norm architecture
            residual = hidden_states
            hidden_states = ln(hidden_states)
            
            # Self-attention
            attn_output, attn_weights, _ = attn_layer(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions
            )
            
            # Residual connection
            hidden_states = residual + attn_output
            
            # FFN
            residual = hidden_states
            hidden_states = hidden_states + ffn(hidden_states)
            
            # Calculate consciousness scores
            if consciousness_state is not None:
                layer_scores = self.consciousness_scorer(hidden_states)
                consciousness_scores.append(layer_scores)
            
            all_hidden_states.append(hidden_states)
            if output_attentions:
                all_attentions.append(attn_weights)
        
        # Performance metrics
        inference_time = time.time() - start_time
        
        # Aggregate consciousness scores
        if consciousness_scores:
            final_consciousness_score = torch.stack(consciousness_scores).mean(0)
        else:
            final_consciousness_score = None
        
        outputs = {
            'hidden_states': hidden_states,
            'all_hidden_states': all_hidden_states,
            'consciousness_score': final_consciousness_score,
            'metrics': {
                'inference_time_ms': inference_time * 1000,
                'sequence_length': hidden_states.shape[1],
                'using_flash_attention': FLASH_ATTENTION_AVAILABLE and self.config.use_flash_attention,
                'gpu_memory_mb': torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
            }
        }
        
        if output_attentions:
            outputs['attentions'] = all_attentions
            
        return hidden_states, outputs

# Benchmark utilities
class AttentionBenchmark:
    """Benchmark different attention implementations"""
    
    @staticmethod
    def benchmark_attention(
        batch_size: int = 4,
        seq_lengths: List[int] = [512, 1024, 2048, 4096, 8192],
        hidden_size: int = 768,
        num_heads: int = 12
    ) -> Dict[str, Any]:
        """Benchmark attention performance"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        results = {}
        
        for seq_len in seq_lengths:
            # Create random input
            input_tensor = torch.randn(
                batch_size, seq_len, hidden_size, 
                device=device, dtype=torch.bfloat16
            )
            
            # Test Flash Attention
            if FLASH_ATTENTION_AVAILABLE:
                config = AttentionConfig(
                    hidden_size=hidden_size,
                    num_attention_heads=num_heads,
                    use_flash_attention=True,
                    use_bf16=True
                )
                
                model = OptimizedMultiHeadAttention(config).to(device)
                model.eval()
                
                # Warmup
                for _ in range(5):
                    with torch.no_grad():
                        _ = model(input_tensor)
                
                # Benchmark
                torch.cuda.synchronize()
                start = time.time()
                
                with torch.no_grad():
                    for _ in range(10):
                        _ = model(input_tensor)
                        
                torch.cuda.synchronize()
                flash_time = (time.time() - start) / 10
                
                results[f'flash_seq{seq_len}'] = {
                    'time_ms': flash_time * 1000,
                    'memory_mb': torch.cuda.max_memory_allocated() / 1024 / 1024
                }
            
            # Test PyTorch native
            if PYTORCH_2_AVAILABLE:
                config = AttentionConfig(
                    hidden_size=hidden_size,
                    num_attention_heads=num_heads,
                    use_flash_attention=False,
                    use_bf16=True
                )
                
                model = OptimizedMultiHeadAttention(config).to(device)
                model.eval()
                
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.time()
                
                with torch.no_grad():
                    for _ in range(10):
                        _ = model(input_tensor)
                        
                torch.cuda.synchronize()
                native_time = (time.time() - start) / 10
                
                results[f'native_seq{seq_len}'] = {
                    'time_ms': native_time * 1000,
                    'memory_mb': torch.cuda.max_memory_allocated() / 1024 / 1024
                }
                
                # Calculate speedup
                if FLASH_ATTENTION_AVAILABLE:
                    speedup = native_time / flash_time
                    results[f'speedup_seq{seq_len}'] = speedup
        
        return results