"""
Remora-style Model for Basecalling
===================================
This module implements a Remora-inspired architecture that can be used 
as an alternative to Melchior for nanopore basecalling tasks.

Key differences from Melchior:
- Uses CNN + Transformer encoder stack (Remora-style) instead of Mamba+Transformer hybrid
- Simplified architecture focusing on proven Remora design patterns
- Maintains compatibility with Melchior's training pipeline and data format
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import math
from typing import Optional, Tuple, Dict, Any
import os


# ============================================================================
# Model Components (Remora-style)
# ============================================================================

class SignalEmbedding(nn.Module):
    """
    1D CNN embedding layer for raw signal input.
    Similar to Remora's signal processing front-end.
    
    Args:
        in_chans: Number of input channels (typically 1 for raw signal)
        embed_dim: Output embedding dimension
        kernel_sizes: List of kernel sizes for convolutional layers
    """
    def __init__(
        self,
        in_chans: int = 1,
        embed_dim: int = 256,
        kernel_sizes: list = [5, 5, 5],
        dropout: float = 0.1
    ):
        super().__init__()
        
        layers = []
        in_dim = in_chans
        
        for i, kernel_size in enumerate(kernel_sizes):
            out_dim = embed_dim // (2 ** (len(kernel_sizes) - i - 1))
            layers.extend([
                nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, 
                         padding=kernel_size//2, bias=False),
                nn.BatchNorm1d(out_dim),
                nn.GELU(),
                nn.Dropout(dropout if i < len(kernel_sizes) - 1 else 0)
            ])
            in_dim = out_dim
        
        # Final projection to embed_dim
        if out_dim != embed_dim:
            layers.append(
                nn.Conv1d(out_dim, embed_dim, kernel_size=1, bias=False)
            )
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, seq_len)
        Returns:
            Embedded tensor of shape (batch, seq_len, embed_dim)
        """
        x = self.network(x)  # (B, embed_dim, L)
        return x.transpose(1, 2)  # (B, L, embed_dim)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequence models.
    """
    def __init__(self, embed_dim: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * 
                           (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, 1, embed_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, embed_dim)
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class RemoraEncoderLayer(nn.Module):
    """
    Transformer encoder layer following Remora design patterns.
    Uses pre-normalization for better training stability.
    Supports memory-efficient attention for long sequences.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True
    ):
        super().__init__()
        
        self.norm_first = norm_first
        
        # Use scaled_dot_product_attention directly for memory efficiency
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
    def forward(self, x: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            attn_mask: Attention mask (optional)
        """
        if self.norm_first:
            # Pre-norm architecture (better for deep networks)
            x = x + self._sa_block(self.norm1(x), attn_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            # Post-norm architecture (traditional)
            x = self.norm1(x + self._sa_block(x, attn_mask))
            x = self.norm2(x + self._ff_block(x))
        
        return x
    
    def _sa_block(self, x: torch.Tensor, 
                  attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, L, D = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        
        # Use scaled_dot_product_attention with is_causal=False for bidirectional attention
        attn_out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False
        )
        
        # Reshape and project output
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        return self.dropout1(self.out_proj(attn_out))
    
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


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
