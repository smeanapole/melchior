#!/usr/bin/env python3
"""
Training and Inference Script for Remora-style Basecaller
==========================================================

This script provides training and inference capabilities for the Remora-style
basecalling model, serving as an alternative to Melchior.

Usage:
    # Training
    python train_remora.py --epochs 20 --batch_size 32 --lr 6e-4
    
    # Inference
    python train_remora.py --mode inference --checkpoint_path models/remora/last.ckpt
    
    # Compare with Melchior
    python train_remora.py --compare
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from typing import Optional, Union
import re
import subprocess

# Import Remora model
from model.remora_style import (
    RemoraBasecaller, 
    RemoraModule, 
    create_remora_model,
    count_parameters
)

# Import dataset utility (same as Melchior uses)
try:
    from utils.data import MelchiorDataset
except ImportError:
    print("Warning: Could not import MelchiorDataset. Please ensure utils/data.py is available.")
    MelchiorDataset = None

torch.set_float32_matmul_precision('medium')


def train_remora(
    state_dict: Optional[str] = None,
    epochs: int = 20,
    batch_size: int = 12,
    lr: float = 2e-3,
    weight_decay: float = 0.01,
    save_path: str = "models/remora",
    num_gpus: Optional[int] = None,
    model_variant: str = "base",
    accumulate_grad_batches: int = 1,
    dropout: float = 0.1,
    embed_dim: int = 256,
    num_layers: int = 8,
    num_heads: int = 8,
) -> tuple:
    """
    Train a Remora-style basecalling model.
    
    Args:
        state_dict: Path to checkpoint for resuming training
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        save_path: Directory to save model checkpoints
        num_gpus: Number of GPUs to use (None = all available)
        model_variant: Model size variant ('tiny', 'small', 'base', 'large')
        accumulate_grad_batches: Gradient accumulation steps
        dropout: Dropout rate
        embed_dim: Embedding dimension (overrides variant default if specified)
        num_layers: Number of encoder layers (overrides variant default if specified)
        num_heads: Number of attention heads (overrides variant default if specified)
    
    Returns:
        Tuple of (train_loss, val_loss, last_lr)
    """
    
    if MelchiorDataset is None:
        raise ImportError("MelchiorDataset is required for training. Please install required dependencies.")
    
    # Create data loaders
    print("Loading datasets...")
    data_train = MelchiorDataset("data/train_val/rna-train.hdf5")
    data_valid = MelchiorDataset("data/train_val/rna-valid.hdf5")
    
    train_loader = DataLoader(
        data_train, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True
    )
    val_loader = DataLoader(
        data_valid, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=8, 
        pin_memory=True
    )
    
    # Create model
    print(f"Creating Remora model (variant: {model_variant})...")
    model = RemoraModule(
        train_loader=train_loader,
        epochs=epochs,
        in_chans=1,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        lr=lr,
        weight_decay=weight_decay,
        accumulate_grad_batches=accumulate_grad_batches,
        dropout=dropout,
        output_length=420,
        input_length=4096,
        num_classes=5
    )
    
    # Print model info
    total_params = count_parameters(model.model)
    print(f"Model parameters: {total_params:,}")
    
    # Get GPU count
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    if num_gpus > 0:
        print(f"Using {num_gpus} GPU(s)")
    else:
        print("No GPUs available, using CPU")
    
    # Setup callbacks
    os.makedirs(save_path, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        filename='{epoch}-{train_loss:.2f}',
        save_top_k=-1,
        monitor='train_loss',
        save_on_train_epoch_end=True,
        save_last=True,
        every_n_epochs=1,
    )
    
    logger = TensorBoardLogger(save_dir="logs/", name="remora")
    
    # Setup trainer
    strategy = "auto"
    if num_gpus > 1:
        from pytorch_lightning.strategies import DDPStrategy
        strategy = DDPStrategy(find_unused_parameters=False)
    
    trainer = pl.Trainer(
        profiler="simple",
        max_epochs=epochs,
        accelerator='gpu' if num_gpus > 0 else 'cpu',
        devices=num_gpus if num_gpus > 0 else 1,
        precision="bf16-mixed" if num_gpus > 0 else "32",
        callbacks=[checkpoint_callback],
        log_every_n_steps=100,
        accumulate_grad_batches=accumulate_grad_batches,
        logger=logger,
        strategy=strategy,
    )
    
    # Train
    print("Starting training...")
    if state_dict:
        print(f"Resuming from checkpoint: {state_dict}")
        trainer.fit(
            model, 
            train_dataloaders=train_loader, 
            val_dataloaders=val_loader, 
            ckpt_path=state_dict
        )
    else:
        trainer.fit(
            model, 
            train_dataloaders=train_loader, 
            val_dataloaders=val_loader
        )
    
    print("Training complete!")
    
    # Get final metrics
    train_loss = trainer.callback_metrics.get('train_loss', torch.tensor(0.0))
    val_loss = trainer.callback_metrics.get('val_loss', torch.tensor(0.0))
    last_lr = model.lr_schedulers().get_last_lr()[0] if hasattr(model, 'lr_schedulers') else lr
    
    return train_loss.item(), val_loss.item(), last_lr


def run_inference(
    checkpoint_path: str,
    input_data: torch.Tensor,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> torch.Tensor:
    """
    Run inference with a trained Remora model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        input_data: Input signal tensor (batch, 1, seq_len)
        device: Device to run inference on
    
    Returns:
        Log-probabilities tensor (output_length, batch, num_classes)
    """
    # Load model
    model = create_remora_model('base')
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove 'model.' prefix if present
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}, using random weights")
    
    model = model.to(device)
    model.eval()
    
    # Run inference
    with torch.no_grad():
        input_data = input_data.to(device)
        output = model(input_data)
    
    return output


def compare_models():
    """Compare Remora and Melchior model architectures."""
    print("\n" + "="*70)
    print("MODEL COMPARISON: Melchior vs Remora")
    print("="*70)
    
    # Import Melchior for comparison
    try:
        from model.melchior import Melchior
        
        # Create both models
        melchior = Melchior(in_chans=1, embed_dim=768, depth=20)
        remora = create_remora_model('base')
        
        melchior_params = sum(p.numel() for p in melchior.parameters())
        remora_params = count_parameters(remora)
        
        print(f"\n{'Model':<15} {'Parameters':<15} {'Architecture':<40}")
        print("-"*70)
        print(f"{'Melchior':<15} {melchior_params:>12,}  Mamba + Transformer hybrid")
        print(f"{'Remora-base':<15} {remora_params:>12,}  CNN + Transformer encoder stack")
        
        print(f"\nKey Differences:")
        print(f"  • Melchior: Uses MambaVisionMixer (state-space model) alternating with Transformer blocks")
        print(f"  • Remora: Uses standard Transformer encoder layers with pre-normalization")
        print(f"  • Melchior: More complex architecture with bidirectional Mamba")
        print(f"  • Remora: Simpler, more interpretable architecture")
        print(f"  • Both: Support CTC loss for sequence-to-sequence basecalling")
        
        # Test forward pass speed (CPU only for fairness)
        print(f"\nSpeed Comparison (CPU, batch_size=1, seq_len=1024):")
        import time
        
        x = torch.randn(1, 1, 1024)
        
        # Warmup
        _ = melchior(x)
        _ = remora(x)
        
        # Time Melchior
        start = time.time()
        for _ in range(10):
            _ = melchior(x)
        melchior_time = (time.time() - start) / 10 * 1000
        
        # Time Remora
        start = time.time()
        for _ in range(10):
            _ = remora(x)
        remora_time = (time.time() - start) / 10 * 1000
        
        print(f"  • Melchior: {melchior_time:.2f} ms per batch")
        print(f"  • Remora:   {remora_time:.2f} ms per batch")
        print(f"  • Speedup:  {melchior_time/remora_time:.2f}x")
        
    except ImportError as e:
        print(f"Could not import Melchior for comparison: {e}")
        remora = create_remora_model('base')
        print(f"\nRemora-base parameters: {count_parameters(remora):,}")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train and infer with Remora-style basecaller")
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="train", 
                       choices=["train", "inference", "compare"],
                       help="Mode: train, inference, or compare")
    
    # Training arguments
    parser.add_argument("--state_dict", type=str, default=None,
                       help="Path to checkpoint for resuming training")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Training batch size")
    parser.add_argument("--lr", type=float, default=6e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--save_path", type=str, default="models/remora",
                       help="Directory to save checkpoints")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--num_gpus", type=int, default=None,
                       help="Number of GPUs to use (default: all available)")
    
    # Model architecture arguments
    parser.add_argument("--model_variant", type=str, default="base",
                       choices=["tiny", "small", "base", "large"],
                       help="Model size variant")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate")
    parser.add_argument("--embed_dim", type=int, default=None,
                       help="Embedding dimension (overrides variant default)")
    parser.add_argument("--num_layers", type=int, default=None,
                       help="Number of encoder layers (overrides variant default)")
    parser.add_argument("--num_heads", type=int, default=None,
                       help="Number of attention heads (overrides variant default)")
    
    # Inference arguments
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Path to model checkpoint for inference")
    parser.add_argument("--input_file", type=str, default=None,
                       help="Path to input HDF5 file for inference")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        # Get variant defaults
        variants = {
            'tiny': {'embed_dim': 128, 'num_layers': 4, 'num_heads': 4},
            'small': {'embed_dim': 256, 'num_layers': 6, 'num_heads': 8},
            'base': {'embed_dim': 256, 'num_layers': 8, 'num_heads': 8},
            'large': {'embed_dim': 512, 'num_layers': 12, 'num_heads': 16},
        }
        
        config = variants[args.model_variant]
        embed_dim = args.embed_dim if args.embed_dim else config['embed_dim']
        num_layers = args.num_layers if args.num_layers else config['num_layers']
        num_heads = args.num_heads if args.num_heads else config['num_heads']
        
        try:
            train_remora(
                state_dict=args.state_dict,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                save_path=args.save_path,
                num_gpus=args.num_gpus,
                model_variant=args.model_variant,
                accumulate_grad_batches=args.accumulate_grad_batches,
                dropout=args.dropout,
                embed_dim=embed_dim,
                num_layers=num_layers,
                num_heads=num_heads,
            )
        except Exception as e:
            print(f"Training failed: {str(e)}")
            raise
    
    elif args.mode == "inference":
        if not args.checkpoint_path:
            print("Error: --checkpoint_path is required for inference mode")
            exit(1)
        
        # Create dummy input for demonstration
        batch_size = 1
        input_length = 4096
        input_data = torch.randn(batch_size, 1, input_length)
        
        print(f"Running inference with checkpoint: {args.checkpoint_path}")
        output = run_inference(args.checkpoint_path, input_data)
        print(f"Output shape: {output.shape}")
        print("Inference complete!")
    
    elif args.mode == "compare":
        compare_models()
    
    else:
        print(f"Unknown mode: {args.mode}")
        exit(1)


if __name__ == "__main__":
    main()
