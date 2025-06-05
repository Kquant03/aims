# salience_scoring.py - PyTorch Multi-Dimensional Salience Scoring
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class SalienceConfig:
    """Configuration for salience scoring"""
    input_dim: int = 768
    hidden_dim: int = 512
    text_embedding_dim: int = 768
    goal_dim: int = 256
    num_classes: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Weights for final score combination
    uncertainty_weight: float = 0.25
    emotion_weight: float = 0.25
    temporal_weight: float = 0.25
    goal_weight: float = 0.25

class MultiDimensionalSalienceScorer(nn.Module):
    """Complete implementation of multi-dimensional salience scoring"""
    
    def __init__(self, config: SalienceConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        # Uncertainty Reduction Module
        self.uncertainty_encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_dim // 2)
        )
        self.evidence_head = nn.Linear(config.hidden_dim // 2, config.num_classes)
        
        # Emotional Valence Module (PAD model)
        self.emotion_encoder = nn.Sequential(
            nn.Linear(config.text_embedding_dim, config.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2)
        )
        
        # Separate heads for PAD dimensions
        self.valence_head = nn.Sequential(
            nn.Linear(config.hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # [-1, 1] range
        )
        
        self.arousal_head = nn.Sequential(
            nn.Linear(config.hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # [0, 1] range
        )
        
        self.dominance_head = nn.Sequential(
            nn.Linear(config.hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # [0, 1] range
        )
        
        # Temporal Relevance Module
        self.temporal_attention = nn.MultiheadAttention(
            config.hidden_dim, 
            num_heads=8, 
            batch_first=True,
            dropout=0.1
        )
        
        self.temporal_encoder = nn.Sequential(
            nn.Linear(config.input_dim + 1, config.hidden_dim),  # +1 for time feature
            nn.ReLU(),
            nn.LayerNorm(config.hidden_dim)
        )
        
        # Learnable time decay parameter
        self.time_decay_param = nn.Parameter(torch.tensor(0.1))
        
        # Goal Alignment Module
        self.memory_projection = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_dim)
        )
        
        self.goal_projection = nn.Sequential(
            nn.Linear(config.goal_dim, config.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_dim)
        )
        
        self.alignment_scorer = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Cross-dimensional attention for fusion
        self.dimension_attention = nn.MultiheadAttention(
            4, num_heads=1, batch_first=True
        )
        
        # Learnable dimension weights
        self.dimension_weights = nn.Parameter(torch.ones(4) / 4)
        
        # Final scoring network with skip connections
        self.final_scorer = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Move to device
        self.to(self.device)
        
        logger.info(f"Initialized salience scorer on {self.device}")
    
    def compute_uncertainty_score(self, memory_embeddings: torch.Tensor, num_samples: int = 10) -> torch.Tensor:
        """Epistemic uncertainty estimation using MC Dropout"""
        # Ensure on correct device
        memory_embeddings = memory_embeddings.to(self.device)
        
        # Enable dropout for MC sampling
        self.uncertainty_encoder.train()
        self.evidence_head.train()
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                h = self.uncertainty_encoder(memory_embeddings)
                evidence = self.evidence_head(h)
                pred = F.softmax(evidence, dim=-1)
                predictions.append(pred)
        
        # Stack predictions
        predictions = torch.stack(predictions)
        mean_pred = torch.mean(predictions, dim=0)
        
        # Calculate entropy as uncertainty measure
        entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-8), dim=-1)
        
        # Normalize to [0, 1]
        max_entropy = torch.log(torch.tensor(self.config.num_classes, device=self.device))
        uncertainty_score = entropy / max_entropy
        
        # Higher uncertainty = higher salience (more to learn)
        return uncertainty_score
    
    def compute_emotional_score(self, text_embeddings: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """VAD (Valence-Arousal-Dominance) model for emotion scoring"""
        text_embeddings = text_embeddings.to(self.device)
        
        # Encode emotional features
        h = self.emotion_encoder(text_embeddings)
        
        # Get PAD values
        valence = self.valence_head(h).squeeze(-1)  # [-1, 1]
        arousal = self.arousal_head(h).squeeze(-1)  # [0, 1]
        dominance = self.dominance_head(h).squeeze(-1)  # [0, 1]
        
        # Compute emotional intensity using Euclidean distance from neutral
        # Neutral point is (0, 0.5, 0.5) in PAD space
        neutral_point = torch.tensor([0.0, 0.5, 0.5], device=self.device)
        emotional_vector = torch.stack([valence, arousal, dominance], dim=-1)
        
        # Calculate distance from neutral
        intensity = torch.norm(emotional_vector - neutral_point, dim=-1) / torch.sqrt(torch.tensor(3.0))
        
        # Emotional salience combines intensity with absolute valence
        emotional_score = (torch.abs(valence) * 0.4 + arousal * 0.3 + intensity * 0.3)
        
        # Ensure in [0, 1] range
        emotional_score = torch.clamp(emotional_score, 0, 1)
        
        pad_values = {
            'valence': valence,
            'arousal': arousal,
            'dominance': dominance,
            'intensity': intensity
        }
        
        return emotional_score, pad_values
    
    def compute_temporal_score(self, memory_embeddings: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """Temporal attention with learnable decay"""
        memory_embeddings = memory_embeddings.to(self.device)
        timestamps = timestamps.to(self.device)
        
        batch_size = memory_embeddings.size(0)
        
        # Normalize timestamps to [0, 1]
        if timestamps.numel() > 0:
            current_time = timestamps.max()
            time_diffs = (current_time - timestamps) / (current_time + 1e-8)
        else:
            time_diffs = torch.zeros(batch_size, device=self.device)
        
        # Reshape time diffs for concatenation
        time_diffs = time_diffs.unsqueeze(-1)
        
        # Combine embeddings with temporal features
        temporal_features = torch.cat([memory_embeddings, time_diffs], dim=-1)
        temporal_encoded = self.temporal_encoder(temporal_features)
        
        # Apply temporal attention (self-attention to find patterns)
        # Add sequence dimension
        temporal_encoded = temporal_encoded.unsqueeze(1)
        
        attn_output, attn_weights = self.temporal_attention(
            temporal_encoded,
            temporal_encoded,
            temporal_encoded
        )
        
        # Remove sequence dimension
        attn_output = attn_output.squeeze(1)
        
        # Exponential decay with learnable parameter
        decay_weights = torch.exp(-torch.abs(self.time_decay_param) * time_diffs.squeeze() * 10)
        
        # Extract attention-based temporal relevance
        temporal_relevance = torch.norm(attn_output, dim=-1) / (torch.sqrt(torch.tensor(self.config.hidden_dim, dtype=torch.float32)))
        
        # Combine decay with attention-based relevance
        temporal_scores = decay_weights * 0.5 + temporal_relevance * 0.5
        
        return torch.clamp(temporal_scores, 0, 1)
    
    def compute_goal_alignment(self, memory_embeddings: torch.Tensor, current_goals: torch.Tensor) -> torch.Tensor:
        """Goal alignment scoring with cosine similarity and learned alignment"""
        memory_embeddings = memory_embeddings.to(self.device)
        current_goals = current_goals.to(self.device)
        
        batch_size = memory_embeddings.size(0)
        
        # Project memories and goals to common space
        mem_proj = self.memory_projection(memory_embeddings)
        goal_proj = self.goal_projection(current_goals)
        
        # Handle different goal tensor shapes
        if goal_proj.dim() == 2:
            # Single goal for all memories
            goal_proj = goal_proj.unsqueeze(0).expand(batch_size, -1, -1)
        elif goal_proj.dim() == 3 and goal_proj.size(0) == 1:
            # Broadcast single goal
            goal_proj = goal_proj.expand(batch_size, -1, -1)
        
        # For each memory, compute alignment with all goals
        alignment_scores = []
        
        for i in range(batch_size):
            mem_i = mem_proj[i:i+1]  # Keep batch dimension
            
            # Expand memory to match number of goals
            num_goals = goal_proj.size(1)
            mem_expanded = mem_i.expand(num_goals, -1)
            
            # Get goals for this memory
            goals_i = goal_proj[i] if goal_proj.size(0) > 1 else goal_proj[0]
            
            # Concatenate for alignment scoring
            combined = torch.cat([mem_expanded, goals_i], dim=-1)
            
            # Compute alignment scores
            scores = self.alignment_scorer(combined).squeeze(-1)
            
            # Average over goals
            avg_score = scores.mean()
            alignment_scores.append(avg_score)
        
        alignment_scores = torch.stack(alignment_scores)
        
        # Also compute direct cosine similarity
        mem_norm = F.normalize(mem_proj, dim=-1)
        goal_norm = F.normalize(goal_proj.mean(dim=1), dim=-1)  # Average goal representation
        
        cosine_sim = F.cosine_similarity(mem_norm, goal_norm, dim=-1)
        
        # Combine learned and geometric alignment
        final_alignment = 0.7 * alignment_scores + 0.3 * (cosine_sim + 1) / 2
        
        return torch.clamp(final_alignment, 0, 1)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Complete forward pass computing all salience dimensions"""
        # Extract inputs and move to device
        memory_embeddings = batch['memory_embeddings'].to(self.device)
        text_embeddings = batch.get('text_embeddings', memory_embeddings).to(self.device)
        timestamps = batch.get('timestamps', torch.zeros(memory_embeddings.size(0))).to(self.device)
        current_goals = batch.get('current_goals', torch.randn(1, self.config.goal_dim)).to(self.device)
        
        batch_size = memory_embeddings.size(0)
        
        # Compute individual scores
        uncertainty_scores = self.compute_uncertainty_score(memory_embeddings)
        emotional_scores, pad_values = self.compute_emotional_score(text_embeddings)
        temporal_scores = self.compute_temporal_score(memory_embeddings, timestamps)
        goal_scores = self.compute_goal_alignment(memory_embeddings, current_goals)
        
        # Stack dimensions [batch_size, 4]
        dimension_scores = torch.stack([
            uncertainty_scores,
            emotional_scores,
            temporal_scores,
            goal_scores
        ], dim=-1)
        
        # Apply learned dimension weights
        dimension_weights_normalized = F.softmax(self.dimension_weights, dim=0)
        weighted_scores = dimension_scores * dimension_weights_normalized
        
        # Use cross-attention for dynamic weighting
        scores_for_attention = dimension_scores.unsqueeze(1)  # [batch, 1, 4]
        
        with torch.no_grad():
            # Self-attention to find important dimension combinations
            fused_scores, attention_weights = self.dimension_attention(
                scores_for_attention,
                scores_for_attention,
                scores_for_attention
            )
        
        # Final salience score computation
        # Combine weighted average with learned fusion
        weighted_avg = weighted_scores.sum(dim=-1, keepdim=True)  # [batch, 1]
        
        # Pass all dimension scores through final scorer
        final_scores = self.final_scorer(dimension_scores)
        
        # Combine approaches
        combined_scores = 0.6 * final_scores + 0.4 * weighted_avg
        
        return {
            'salience_scores': combined_scores.squeeze(-1),
            'dimension_scores': dimension_scores,
            'dimension_weights': dimension_weights_normalized,
            'attention_weights': attention_weights.squeeze(1) if attention_weights is not None else None,
            'emotional_components': pad_values,
            'weighted_scores': weighted_scores
        }
    
    def compute_batch_salience(self, memories: List[Dict[str, Any]], goals: Optional[torch.Tensor] = None) -> List[Dict[str, float]]:
        """Compute salience scores for a batch of memories"""
        self.eval()  # Set to evaluation mode
        
        # Prepare batch
        memory_embeds = []
        text_embeds = []
        timestamps = []
        
        for memory in memories:
            # Get embeddings
            mem_embed = memory.get('embedding', np.random.randn(self.config.input_dim))
            text_embed = memory.get('text_embedding', mem_embed)
            
            # Convert to tensors if needed
            if isinstance(mem_embed, list):
                mem_embed = np.array(mem_embed)
            if isinstance(text_embed, list):
                text_embed = np.array(text_embed)
                
            memory_embeds.append(mem_embed)
            text_embeds.append(text_embed)
            
            # Handle timestamp
            ts = memory.get('timestamp', datetime.now())
            if isinstance(ts, datetime):
                ts = ts.timestamp()
            timestamps.append(ts)
        
        # Convert to tensors
        memory_embeddings = torch.tensor(np.array(memory_embeds), dtype=torch.float32)
        text_embeddings = torch.tensor(np.array(text_embeds), dtype=torch.float32)
        timestamps_tensor = torch.tensor(timestamps, dtype=torch.float32)
        
        # Default goals if not provided
        if goals is None:
            goals = torch.randn(1, self.config.goal_dim)
        
        # Create batch
        batch = {
            'memory_embeddings': memory_embeddings,
            'text_embeddings': text_embeddings,
            'timestamps': timestamps_tensor,
            'current_goals': goals
        }
        
        # Compute scores
        with torch.no_grad():
            results = self.forward(batch)
        
        # Format results
        salience_results = []
        for i in range(len(memories)):
            salience_results.append({
                'overall_salience': float(results['salience_scores'][i]),
                'uncertainty': float(results['dimension_scores'][i, 0]),
                'emotional': float(results['dimension_scores'][i, 1]),
                'temporal': float(results['dimension_scores'][i, 2]),
                'goal_alignment': float(results['dimension_scores'][i, 3]),
                'emotional_details': {
                    'valence': float(results['emotional_components']['valence'][i]),
                    'arousal': float(results['emotional_components']['arousal'][i]),
                    'dominance': float(results['emotional_components']['dominance'][i]),
                    'intensity': float(results['emotional_components']['intensity'][i])
                }
            })
        
        return salience_results