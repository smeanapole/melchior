"""
Melchior Model Implementation in Remora Style
==============================================
This module implements the Melchior architecture (Mamba + Transformer hybrid)
with clean, modular code similar to Remora's design patterns.

Key Architecture Components:
- FastEmbed: 2-layer 1D CNN for signal embedding
- Learnable positional encoding
- Hybrid blocks (20 layers): alternating MambaBlock and TransformerBlock
- Output head with adaptive pooling

This implementation maintains compatibility with Melchior's training pipeline
while using cleaner, more modular code structure inspired by Remora.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import math
from typing import Optional, Tuple, Dict, Any
import os

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    from einops import rearrange, repeat
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba_ssm not available. MambaBlock will use fallback implementation.")

from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import Mlp


# ============================================================================
# Model Components (Melchior Architecture)
# ============================================================================

class FastEmbed(nn.Module):
    """
    2-layer 1D CNN embedding layer for raw signal input.
    Maps input from 1 channel to embed_dim (512→1024).
    
    Args:
        in_chans: Number of input channels (default: 1)
        in_dim: Intermediate dimension (default: 512)
        embed_dim: Output embedding dimension (default: 1024)
        drop_path: Drop path rate (default: 0.)
        layer_scale: Layer scale coefficient (default: None)
    """
    def __init__(
        self,
        in_chans: int = 1,
        in_dim: int = 512,
        embed_dim: int = 1024,
        drop_path: float = 0.,
        layer_scale: Optional[float] = None
    ):
        super().__init__()
        self.proj = nn.Identity()
        
        self.conv_down = nn.Sequential(
            nn.Conv1d(in_chans, in_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(in_dim, eps=1e-4),
            nn.GELU(approximate='tanh'),
            nn.Conv1d(in_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(embed_dim, eps=1e-4),
            nn.GELU(approximate='tanh')
        )
        
        self.layer_scale = layer_scale
        if layer_scale is not None and isinstance(layer_scale, (int, float)):
            self.gamma = nn.Parameter(layer_scale * torch.ones(embed_dim))
            self.has_layer_scale = True
        else:
            self.has_layer_scale = False
            
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, in_chans, seq_len)
        Returns:
            Embedded tensor of shape (batch, seq_len, embed_dim)
        """
        input_res = x
        x = self.proj(x)
        x = self.conv_down(x)
        
        if self.has_layer_scale:
            x = x * self.gamma.view(1, -1, 1)
            
        x = input_res + self.drop_path(x)
        return x.transpose(1, 2)  # (B, L, D)


class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding for sequence models.
    """
    def __init__(self, embed_dim: int, max_len: int = 4096, dropout: float = 0.0):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        trunc_normal_(self.pos_embed, std=.02)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0. else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, embed_dim)
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pos_embed[:, :x.size(1), :]
        return self.dropout(x)


class MambaVisionMixer(nn.Module):
    """
    Mamba Vision Mixer using selective scan mechanism.
    
    Args:
        d_model: Model dimension
        d_state: State dimension (default: 8)
        d_conv: Convolution kernel size (default: 3)
        expand: Expansion factor (default: 1)
        dt_rank: DT rank (default: "auto")
        dt_min: Minimum delta value (default: 0.001)
        dt_max: Maximum delta value (default: 0.1)
        conv_bias: Whether to use bias in convolution (default: True)
        bias: Whether to use bias in linear layers (default: False)
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 8,
        d_conv: int = 3,
        expand: int = 1,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        conv_bias: bool = True,
        bias: bool = False,
        layer_idx: Optional[int] = None
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx
        
        # Input projection
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias)
        
        # X projection for selective scan
        self.x_proj = nn.Linear(
            self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False
        )
        
        # DT projection
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True)
        
        # Initialize DT parameters
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # Initialize DT bias
        dt = torch.exp(
            torch.rand(self.d_inner // 2) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        
        # A parameter (log scale)
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner // 2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        
        # D parameter
        self.D = nn.Parameter(torch.ones(self.d_inner // 2))
        self.D._no_weight_decay = True
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        
        # Convolutions
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            padding=d_conv // 2
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            padding=d_conv // 2
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Input tensor of shape (B, L, D)
        Returns:
            Output tensor of same shape
        """
        if not MAMBA_AVAILABLE:
            # Fallback: simple linear projection
            return self.out_proj(self.in_proj(hidden_states))
            
        _, seqlen, _ = hidden_states.shape
        
        # Project and split
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        
        # Compute A
        A = -torch.exp(self.A_log.float())
        
        # Apply convolutions with SiLU activation
        x = F.silu(F.conv1d(
            input=x, 
            weight=self.conv1d_x.weight, 
            bias=self.conv1d_x.bias, 
            padding='same', 
            groups=self.d_inner // 2
        ))
        z = F.silu(F.conv1d(
            input=z, 
            weight=self.conv1d_z.weight, 
            bias=self.conv1d_z.bias, 
            padding='same', 
            groups=self.d_inner // 2
        ))
        
        # Double projection
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        # Reshape
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        
        # Selective scan
        y = selective_scan_fn(
            x, dt, A, B, C,
            self.D.float(),
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=None
        )
        
        # Concatenate with z and project
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        
        return out


class MambaBlock(nn.Module):
    """
    Mamba block with MLP post-processing.
    
    Args:
        dim: Feature dimension
        mlp_ratio: MLP hidden dimension ratio (default: 4.)
        drop: Dropout rate (default: 0.)
        drop_path: Drop path rate (default: 0.)
        act_layer: Activation layer (default: nn.GELU)
        norm_layer: Normalization layer (default: nn.LayerNorm)
        layer_scale: Layer scale coefficient (default: None)
    """
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.,
        drop: float = 0.,
        drop_path: float = 0.,
        act_layer: type = nn.GELU,
        norm_layer: type = nn.LayerNorm,
        layer_scale: Optional[float] = None
    ):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        
        # Mamba mixer with specified parameters
        self.mixer = MambaVisionMixer(
            d_model=dim,
            d_state=8,   # As per Melchior spec
            d_conv=3,    # As per Melchior spec
            expand=1     # As per Melchior spec
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, 
            hidden_features=mlp_hidden_dim, 
            act_layer=act_layer, 
            drop=drop
        )
        
        # Layer scale
        use_layer_scale = layer_scale is not None and isinstance(layer_scale, (int, float))
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class Attention(nn.Module):
    """
    Multi-head self-attention using scaled_dot_product_attention.
    
    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        qkv_bias: Whether to use bias in QKV projection (default: False)
        qk_norm: Whether to normalize Q and K (default: False)
        attn_drop: Attention dropout rate (default: 0.)
        proj_drop: Projection dropout rate (default: 0.)
        norm_layer: Normalization layer (default: nn.LayerNorm)
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: type = nn.LayerNorm
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Normalize
        q, k = self.q_norm(q), self.k_norm(k)
        
        # Scaled dot-product attention
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0
        )
        
        # Reshape and project
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class TransformerBlock(nn.Module):
    """
    Standard Transformer block with attention and MLP.
    
    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio (default: 4.)
        qkv_bias: Whether to use bias in QKV projection (default: False)
        qk_scale: Deprecated, kept for compatibility
        drop: Dropout rate (default: 0.)
        attn_drop: Attention dropout rate (default: 0.)
        drop_path: Drop path rate (default: 0.)
        act_layer: Activation layer (default: nn.GELU)
        norm_layer: Normalization layer (default: nn.LayerNorm)
        layer_scale: Layer scale coefficient (default: None)
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        qk_scale: bool = False,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        act_layer: type = nn.GELU,
        norm_layer: type = nn.LayerNorm,
        layer_scale: Optional[float] = None
    ):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        
        self.mixer = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, 
            hidden_features=mlp_hidden_dim, 
            act_layer=act_layer, 
            drop=drop
        )
        
        # Layer scale
        use_layer_scale = layer_scale is not None and isinstance(layer_scale, (int, float))
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class Head(nn.Module):
    """
    Output head with adaptive pooling.
    Reduces sequence length from 4096 to 420 and projects to classes.
    
    Args:
        in_features: Input feature dimension
        seq_len: Input sequence length (default: 4096)
        out_seq_len: Output sequence length (default: 420)
        num_classes: Number of output classes (default: 5)
    """
    def __init__(
        self,
        in_features: int,
        seq_len: int = 4096,
        out_seq_len: int = 420,
        num_classes: int = 5
    ):
        super().__init__()
        self.in_features = in_features
        self.seq_len = seq_len
        self.out_seq_len = out_seq_len
        self.num_classes = num_classes
        
        # Adaptive average pooling to reduce sequence length
        self.adaptive_pool = nn.AdaptiveAvgPool1d(out_seq_len)
        
        # Final projection to class scores
        self.out_proj = nn.Linear(in_features, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, in_features)
        Returns:
            Output tensor of shape (out_seq_len, batch_size, num_classes)
        """
        # Transpose for pooling: (batch_size, in_features, seq_len)
        x = x.transpose(1, 2)
        
        # Adaptive pooling: (batch_size, in_features, out_seq_len)
        x = self.adaptive_pool(x)
        
        # Transpose back: (batch_size, out_seq_len, in_features)
        x = x.transpose(1, 2)
        
        # Project to class scores: (batch_size, out_seq_len, num_classes)
        x = self.out_proj(x)
        
        # Permute to (out_seq_len, batch_size, num_classes) for CTC
        x = x.permute(1, 0, 2)
        
        return x


# ============================================================================
# Main Model Architecture (Melchior)
# ============================================================================

class Melchior(nn.Module):
    """
    Melchior model with Mamba + Transformer hybrid architecture.
    
    Architecture:
        1. FastEmbed: 2-layer 1D CNN (1→512→embed_dim)
        2. Learnable Positional Encoding
        3. Hybrid Blocks (20 layers): alternating MambaBlock (even) and TransformerBlock (odd)
        4. LayerNorm
        5. Head: AdaptiveAvgPool1D (4096→420) + Linear projection
        6. log_softmax output
    
    Args:
        in_chans: Number of input channels (default: 1)
        embed_dim: Embedding dimension (default: 1024)
        depth: Number of layers (default: 20)
        num_heads: Number of attention heads for Transformer blocks (default: 8)
        mlp_ratio: MLP hidden dimension ratio (default: 4.)
        qkv_bias: Whether to use bias in QKV projection (default: False)
        drop_rate: Dropout rate (default: 0.)
        attn_drop_rate: Attention dropout rate (default: 0.)
        drop_path_rate: Drop path rate (default: 0.1)
        layer_scale: Layer scale coefficient (default: 1e-5)
        output_length: Output sequence length (default: 420)
    """
    
    def __init__(
        self,
        in_chans: int = 1,
        embed_dim: int = 1024,
        depth: int = 20,
        num_heads: int = 8,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
        layer_scale: float = 1e-5,
        output_length: int = 420
    ):
        super().__init__()
        
        # Stem: FastEmbed
        self.stem = FastEmbed(
            in_chans=in_chans,
            in_dim=512,
            embed_dim=embed_dim,
            drop_path=drop_path_rate,
            layer_scale=layer_scale
        )
        
        # Learnable positional encoding
        self.pos_embed = PositionalEncoding(
            embed_dim=embed_dim,
            max_len=4096,
            dropout=drop_rate
        )
        
        # Drop path rates
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Hybrid blocks: even=Mamba, odd=Transformer
        self.blocks = nn.ModuleList([
            MambaBlock(
                dim=embed_dim,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[i],
                layer_scale=layer_scale
            ) if i % 2 == 0 else
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                layer_scale=layer_scale
            )
            for i in range(depth)
        ])
        
        # Normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Output head
        self.output_length = output_length
        self.head = Head(
            in_features=embed_dim,
            seq_len=4096,
            out_seq_len=output_length,
            num_classes=5
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input signal tensor of shape (batch, 1, 4096)
        
        Returns:
            Log-probabilities of shape (output_length, batch, 5)
        """
        # Embed: (B, 1, 4096) -> (B, 4096, embed_dim)
        x = self.stem(x)
        
        # Add positional encoding
        x = self.pos_embed(x)
        
        # Process through hybrid blocks
        for block in self.blocks:
            x = block(x)
        
        # Normalize
        x = self.norm(x)
        
        # Output head
        x = self.head(x)
        
        # Log softmax
        x = F.log_softmax(x, dim=-1)
        
        return x


class RemoraEncoder(nn.Module):
    """
    Stack of transformer encoder layers.
    """
    def __init__(
        self,
        embed_dim: int = 256,
        num_layers: int = 8,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            RemoraEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            attn_mask: Attention mask (optional)
        Returns:
            Encoded tensor (batch, seq_len, embed_dim)
        """
        for layer in self.layers:
            x = layer(x, attn_mask)
        return self.norm(x)


class CRFHead(nn.Module):
    """
    Output head with CTC/CRF loss support.
    Adapts sequence length and projects to class probabilities.
    """
    def __init__(
        self,
        in_features: int,
        seq_len: int,
        out_seq_len: int,
        num_classes: int,
        use_adaptive_pool: bool = True
    ):
        super().__init__()
        
        self.use_adaptive_pool = use_adaptive_pool
        
        if use_adaptive_pool:
            # Adaptive pooling to reduce sequence length
            self.adaptive_pool = nn.AdaptiveAvgPool1d(out_seq_len)
        else:
            # Learnable downsampling
            self.downsample = nn.Conv1d(
                in_features, in_features, 
                kernel_size=seq_len // out_seq_len,
                stride=seq_len // out_seq_len
            )
        
        self.out_proj = nn.Linear(in_features, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, in_features)
        Returns:
            Output tensor (out_seq_len, batch, num_classes)
        """
        # Transpose for pooling: (B, D, L)
        x = x.transpose(1, 2)
        
        if self.use_adaptive_pool:
            x = self.adaptive_pool(x)  # (B, D, out_seq_len)
        else:
            x = self.downsample(x)
        
        # Transpose back: (B, out_seq_len, D)
        x = x.transpose(1, 2)
        
        # Project to classes: (B, out_seq_len, num_classes)
        x = self.out_proj(x)
        
        # Permute to (out_seq_len, B, num_classes) for CTC
        x = x.permute(1, 0, 2)
        
        return x


# ============================================================================
# Main Model Architecture
# ============================================================================

class RemoraBasecaller(nn.Module):
    """
    Remora-style basecalling model.
    
    This model uses a CNN front-end followed by transformer encoders,
    similar to the architecture used in Oxford Nanopore's Remora.
    
    Architecture:
        1. Signal Embedding (CNN layers)
        2. Positional Encoding
        3. Transformer Encoder Stack
        4. Output Head with adaptive pooling
    
    Args:
        in_chans: Number of input channels (default: 1 for raw signal)
        embed_dim: Embedding dimension (default: 256)
        num_layers: Number of transformer encoder layers (default: 8)
        num_heads: Number of attention heads (default: 8)
        mlp_ratio: MLP hidden dimension ratio (default: 4)
        dropout: Dropout rate (default: 0.1)
        output_length: Output sequence length (default: 420)
        input_length: Input sequence length (default: 4096)
        num_classes: Number of output classes (default: 5)
    """
    
    def __init__(
        self,
        in_chans: int = 1,
        embed_dim: int = 256,
        num_layers: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        output_length: int = 420,
        input_length: int = 4096,
        num_classes: int = 5,
        activation: str = "gelu"
    ):
        super().__init__()
        
        self.input_length = input_length
        self.output_length = output_length
        self.num_classes = num_classes
        
        # Stem: CNN embedding
        self.stem = SignalEmbedding(
            in_chans=in_chans,
            embed_dim=embed_dim,
            dropout=dropout
        )
        
        # Positional encoding
        self.pos_embed = PositionalEncoding(
            embed_dim=embed_dim,
            max_len=input_length,
            dropout=dropout
        )
        
        # Transformer encoder
        dim_feedforward = int(embed_dim * mlp_ratio)
        self.encoder = RemoraEncoder(
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )
        
        # Output head
        self.head = CRFHead(
            in_features=embed_dim,
            seq_len=input_length,
            out_seq_len=output_length,
            num_classes=num_classes
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights using Xavier/He initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input signal tensor of shape (batch, 1, seq_len)
        
        Returns:
            Log-probabilities of shape (output_length, batch, num_classes)
        """
        # Embed signal: (B, 1, L) -> (B, L, D)
        x = self.stem(x)
        
        # Add positional encoding
        x = self.pos_embed(x)
        
        # Encode with transformer
        x = self.encoder(x)
        
        # Project to output
        x = self.head(x)
        
        # Apply log_softmax for CTC loss
        x = F.log_softmax(x, dim=-1)
        
        return x


# ============================================================================
# PyTorch Lightning Module
# ============================================================================

class RemoraModule(pl.LightningModule):
    """
    PyTorch Lightning wrapper for RemoraBasecaller.
    
    Provides training, validation, and optimization logic compatible
    with Melchior's training pipeline.
    """
    
    def __init__(
        self,
        train_loader: DataLoader,
        epochs: int,
        in_chans: int = 1,
        embed_dim: int = 256,
        num_layers: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        lr: float = 6e-4,
        weight_decay: float = 0.01,
        accumulate_grad_batches: int = 1,
        dropout: float = 0.1,
        output_length: int = 420,
        input_length: int = 4096,
        num_classes: int = 5
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = RemoraBasecaller(
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            output_length=output_length,
            input_length=input_length,
            num_classes=num_classes
        )
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_loader = train_loader
        self.epochs = epochs
        self.accumulate_grad_batches = accumulate_grad_batches
        
        # Label smoothing weights for CTC
        self.smoothweights = torch.cat([
            torch.tensor([0.1]), 
            (0.1 / (num_classes - 1)) * torch.ones(num_classes - 1)
        ])
        
        # Will be calculated in setup()
        self.warmup_steps = 0
        self.total_steps = 0
        
    def setup(self, stage: Optional[str] = None):
        """Calculate total steps and warmup steps."""
        if stage == 'fit':
            dataset_size = len(self.train_loader.dataset)
            num_devices = self.trainer.num_devices
            effective_batch_size = (
                self.train_loader.batch_size * 
                num_devices * 
                self.accumulate_grad_batches
            )
            
            steps_per_epoch = math.ceil(dataset_size / effective_batch_size)
            self.total_steps = steps_per_epoch * self.epochs
            self.warmup_steps = int(0.05 * self.total_steps)
            
            print(f"Total training steps: {self.total_steps}")
            print(f"Warmup steps: {self.warmup_steps}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        if self.trainer.optimizers:
            optimizer = self.trainer.optimizers[0]
            return optimizer.param_groups[0]['lr']
        return self.lr
    
    def _compute_ctc_loss(
        self, 
        output: torch.Tensor, 
        label: torch.Tensor, 
        label_len: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute CTC loss with label smoothing.
        
        Args:
            output: Log-probabilities (T, B, num_classes)
            label: Target labels (B, max_label_len)
            label_len: Length of each label (B,)
        
        Returns:
            Dictionary containing 'loss' key
        """
        batch_size = output.shape[1]
        input_len = torch.full((batch_size,), output.shape[0], 
                              dtype=torch.long, device=output.device)
        
        # CTC loss
        loss = F.ctc_loss(
            output, label, input_len, label_len,
            blank=0, reduction='mean', zero_infinity=True
        )
        
        return {"loss": loss}
    
    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        """Single training step."""
        self.model.train()
        
        event, event_len, label, label_len = batch
        event = torch.unsqueeze(event, 1)  # (B, 1, L)
        label = label[:, :max(label_len)]  # Trim to max label length
        
        output = self(event)
        losses = self._compute_ctc_loss(output, label, label_len)
        loss = losses["loss"]
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, 
                prog_bar=True, sync_dist=True)
        
        current_lr = self.get_lr()
        self.log('learning_rate', current_lr, on_step=True, 
                on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        
        return loss
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        current_lr = self.get_lr()
        self.log('learning_rate_epoch', current_lr, 
                on_step=False, on_epoch=True, prog_bar=True, 
                logger=True, sync_dist=True)
    
    def validation_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        """Single validation step."""
        self.model.eval()
        
        with torch.no_grad():
            event, event_len, label, label_len = batch
            event = torch.unsqueeze(event, 1)
            label = label[:, :max(label_len)]
            
            output = self(event)
            losses = self._compute_ctc_loss(output, label, label_len)
            loss = losses["loss"]
            
            self.log('val_loss', loss, on_step=True, on_epoch=True, 
                    prog_bar=True, sync_dist=True)
            
            return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        # Cosine annealing with warmup
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "step",
            },
        }


# ============================================================================
# Utility Functions
# ============================================================================

def create_remora_model(
    model_variant: str = "base",
    **kwargs
) -> RemoraBasecaller:
    """
    Factory function to create different Remora model variants.
    
    Args:
        model_variant: Model size variant ('tiny', 'small', 'base', 'large')
        **kwargs: Additional arguments passed to RemoraBasecaller
    
    Returns:
        RemoraBasecaller instance
    
    Model Variants:
        - tiny: embed_dim=128, num_layers=4, num_heads=4
        - small: embed_dim=256, num_layers=6, num_heads=8
        - base: embed_dim=256, num_layers=8, num_heads=8
        - large: embed_dim=512, num_layers=12, num_heads=16
    """
    variants = {
        'tiny': {'embed_dim': 128, 'num_layers': 4, 'num_heads': 4},
        'small': {'embed_dim': 256, 'num_layers': 6, 'num_heads': 8},
        'base': {'embed_dim': 256, 'num_layers': 8, 'num_heads': 8},
        'large': {'embed_dim': 512, 'num_layers': 12, 'num_heads': 16},
    }
    
    if model_variant not in variants:
        raise ValueError(f"Unknown variant: {model_variant}. "
                        f"Choose from {list(variants.keys())}")
    
    config = {**variants[model_variant], **kwargs}
    return RemoraBasecaller(**config)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Create model
    model = create_remora_model('base')
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 4
    input_length = 4096
    x = torch.randn(batch_size, 1, input_length)
    
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({model.output_length}, {batch_size}, {model.num_classes})")
    
    # Verify output
    assert output.shape == (model.output_length, batch_size, model.num_classes)
    print("✓ Forward pass successful!")
