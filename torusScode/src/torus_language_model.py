# torus_language_model.py
"""
Complete Torus Fractal Language Model
Combines geometry, attention, and fractal hierarchy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List, Tuple
from torus_geometry import TorusLattice, TorusPosition, TorusEmbedding
from torus_tokenizer import TorusTokenizer
from torus_attention import TorusTransformerLayer

class TorusFractalLM(nn.Module):
    """
    Language model that maps tokens to torus surface and uses
    geometric principles for understanding and generation
    """
    
    def __init__(self, 
                 vocab_size: int = 10000,
                 embed_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 ff_dim: int = 2048,
                 max_seq_length: int = 512,
                 R: float = 3.0,
                 r: float = 1.0,
                 num_lattice_levels: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length
        self.R = R
        self.r = r
        
        # Initialize components
        self.tokenizer = TorusTokenizer(R, r)
        self.lattice = TorusLattice(R, r, num_lattice_levels)
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_encoder = TorusPositionalEncoding(embed_dim, R, r)
        
        # Transformer layers with torus attention
        self.layers = nn.ModuleList([
            TorusTransformerLayer(embed_dim, num_heads, ff_dim, R, r, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        
        # Fractal hierarchy connections
        self.level_connections = nn.ModuleDict({
            f"level_{i}": nn.Linear(embed_dim, embed_dim)
            for i in range(num_lattice_levels)
        })
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def map_tokens_to_lattice(self, tokens: List[str]) -> torch.Tensor:
        """Map tokens to torus positions"""
        positions = []
        for token in tokens:
            pos = self.tokenizer.get_position(token)
            positions.append([pos.u, pos.v])
        
        return torch.tensor(positions, dtype=torch.float32)
    
    def forward(self, 
                input_ids: torch.Tensor,
                positions: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                return_lattice_state: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model
        
        Args:
            input_ids: [batch_size, seq_len] token indices
            positions: [batch_size, seq_len, 2] torus positions (optional)
            mask: [batch_size, seq_len] attention mask
            return_lattice_state: whether to return lattice activations
            
        Returns:
            Dictionary with 'logits' and optionally 'lattice_state'
        """
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        x = self.token_embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        
        # Add torus positional encoding
        if positions is None:
            # Generate positions based on token semantics
            # (In practice, you'd map tokens to positions using the tokenizer)
            positions = torch.rand(batch_size, seq_len, 2) * 2 * np.pi
        
        x = self.position_encoder(x, positions)
        x = self.dropout(x)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, positions, mask)
        
        x = self.layer_norm(x)
        
        # Project to vocabulary
        logits = self.output_projection(x)
        
        output = {'logits': logits}
        
        # Optionally compute lattice state
        if return_lattice_state:
            lattice_state = self.compute_lattice_state(x, positions)
            output['lattice_state'] = lattice_state
        
        return output
    
    def compute_lattice_state(self, 
                             hidden_states: torch.Tensor,
                             positions: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Project hidden states onto lattice at different levels
        This creates the fractal representation
        """
        batch_size = hidden_states.shape[0]
        lattice_state = {}
        
        for level in range(len(self.lattice.particles)):
            # Get particles at this level
            particles = self.lattice.particles[level]
            level_size = len(particles)
            
            # Initialize activations for this level
            activations = torch.zeros(batch_size, level_size, self.embed_dim)
            
            # Project hidden states to lattice particles
            for idx, (coords, particle) in enumerate(particles.items()):
                # Compute influence of each token on this particle
                particle_pos = torch.tensor([particle.position.u, particle.position.v])
                
                # Distance from each token to this particle
                distances = self.compute_torus_distances(positions, particle_pos)
                
                # Weight by inverse distance (closer tokens have more influence)
                weights = F.softmax(-distances / 0.5, dim=-1)  # [batch_size, seq_len]
                
                # Weighted sum of hidden states
                particle_activation = torch.matmul(
                    weights.unsqueeze(1), 
                    hidden_states
                ).squeeze(1)  # [batch_size, embed_dim]
                
                # Apply level-specific transformation
                particle_activation = self.level_connections[f"level_{level}"](particle_activation)
                
                activations[:, idx] = particle_activation
            
            lattice_state[level] = activations
        
        return lattice_state
    
    def compute_torus_distances(self, 
                               positions1: torch.Tensor, 
                               position2: torch.Tensor) -> torch.Tensor:
        """Compute distances on torus between multiple positions and a single position"""
        u1, v1 = positions1[:, :, 0], positions1[:, :, 1]
        u2, v2 = position2[0], position2[1]
        
        # Compute wrapped differences
        du = torch.min(torch.abs(u1 - u2), 2*np.pi - torch.abs(u1 - u2))
        dv = torch.min(torch.abs(v1 - v2), 2*np.pi - torch.abs(v1 - v2))
        
        # Geodesic distance
        ds_major = self.R * du
        ds_minor = self.r * dv
        
        return torch.sqrt(ds_major**2 + ds_minor**2)
    
    def generate(self, 
                prompt: str,
                max_length: int = 50,
                temperature: float = 1.0,
                top_k: int = 50) -> str:
        """Generate text starting from a prompt"""
        self.eval()
        
        # Tokenize prompt
        tokens = self.tokenizer.tokenize(prompt)
        token_ids = [self.tokenizer.vocab.get(t, 0) for t in tokens]
        
        # Convert to tensor
        input_ids = torch.tensor([token_ids])
        
        generated = []
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get model predictions
                output = self.forward(input_ids)
                logits = output['logits']
                
                # Get next token logits
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                generated.append(next_token.item())
                
                # Stop if we hit end token
                if next_token.item() == self.tokenizer.vocab.get('<END>', 1):
                    break
        
        # Decode generated tokens
        # (This is simplified - in practice you'd have proper decoding)
        generated_text = ' '.join([str(t) for t in generated])
        
        return prompt + ' ' + generated_text

class TorusPositionalEncoding(nn.Module):
    """Learnable positional encoding based on torus coordinates"""
    
    def __init__(self, embed_dim: int, R: float = 3.0, r: float = 1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.R = R
        self.r = r
        
        # Fourier features for encoding positions
        self.u_frequencies = nn.Parameter(torch.randn(embed_dim // 4))
        self.v_frequencies = nn.Parameter(torch.randn(embed_dim // 4))
        
        # Learnable projection
        self.projection = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to embeddings
        
        Args:
            x: [batch_size, seq_len, embed_dim] embeddings
            positions: [batch_size, seq_len, 2] torus positions (u, v)
        """
        batch_size, seq_len, _ = x.shape
        
        u = positions[:, :, 0]  # [batch_size, seq_len]
        v = positions[:, :, 1]
        
        # Create Fourier features
        u_features = torch.cat([
            torch.sin(u.unsqueeze(-1) * self.u_frequencies),
            torch.cos(u.unsqueeze(-1) * self.u_frequencies)
        ], dim=-1)  # [batch_size, seq_len, embed_dim//2]
        
        v_features = torch.cat([
            torch.sin(v.unsqueeze(-1) * self.v_frequencies),
            torch.cos(v.unsqueeze(-1) * self.v_frequencies)
        ], dim=-1)  # [batch_size, seq_len, embed_dim//2]
        
        # Combine features
        pos_encoding = torch.cat([u_features, v_features], dim=-1)
        pos_encoding = self.projection(pos_encoding)
        
        return x + pos_encoding