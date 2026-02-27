"""
train.py

CLI entry-point for training the Medical-NER model.

Defaults are read from config/config.yaml. Any CLI flag overrides the
corresponding YAML value.

Usage examples:
    python scripts/train.py
    python scripts/train.py --lr 3e-5 --epochs 5 --batch-size 32
    python scripts/train.py --config config/config.yaml --wandb
"""

from __future__ import annotations

import argparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train BiomedBERT for medical NER on BC5CDR",
    )

    p.add_argument("--config", type=str, default="config/config.yaml",
                   help="Path to YAML config file (default: config/config.yaml)")

    # The following flags override values loaded from the YAML file.
    p.add_argument("--lr", type=float, default=None,
                   help="Peak learning rate for AdamW")
    p.add_argument("--weight-decay", type=float, default=None,
                   help="AdamW weight decay")
    p.add_argument("--warmup-steps", type=int, default=None,
                   help="Number of warmup steps for linear scheduler")
    p.add_argument("--epochs", type=int, default=None,
                   help="Max training epochs")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Per-device train batch size")
    p.add_argument("--max-length", type=int, default=None,
                   help="Max subword sequence length")
    p.add_argument("--grad-accum", type=int, default=None,
                   help="Gradient accumulation steps")
    p.add_argument("--no-fp16", action="store_true",
                   help="Disable mixed-precision training")
    p.add_argument("--patience", type=int, default=None,
                   help="Early stopping patience on val F1")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Directory for checkpoints")
    p.add_argument("--logging-dir", type=str, default=None,
                   help="Directory for TensorBoard/CSV logs")
    p.add_argument("--wandb", action="store_true",
                   help="Enable Weights & Biases logging")
    p.add_argument("--wandb-project", type=str, default=None,
                   help="W&B project name")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Import here so --help is fast and doesn't load torch/transformers.
    from src.training.trainer import TrainConfig, train

    # Load defaults from YAML, then apply any CLI overrides.
    cfg = TrainConfig.from_yaml(args.config)

    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.warmup_steps is not None:
        cfg.warmup_steps = args.warmup_steps
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.max_length is not None:
        cfg.max_length = args.max_length
    if args.grad_accum is not None:
        cfg.gradient_accumulation_steps = args.grad_accum
    if args.no_fp16:
        cfg.fp16 = False
    if args.patience is not None:
        cfg.early_stopping_patience = args.patience
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.logging_dir is not None:
        cfg.logging_dir = args.logging_dir
    if args.wandb:
        cfg.use_wandb = True
    if args.wandb_project is not None:
        cfg.wandb_project = args.wandb_project
    if args.seed is not None:
        cfg.seed = args.seed

    print("=" * 55)
    print("  Medical-NER Training")
    print("=" * 55)
    print(f"  config       : {args.config}")
    print(f"  lr           : {cfg.learning_rate}")
    print(f"  epochs       : {cfg.epochs}")
    print(f"  batch_size   : {cfg.batch_size}")
    print(f"  warmup_steps : {cfg.warmup_steps}")
    print(f"  patience     : {cfg.early_stopping_patience}")
    print(f"  fp16         : {cfg.fp16}")
    print(f"  output_dir   : {cfg.output_dir}")
    print(f"  wandb        : {cfg.use_wandb}")
    print(f"  seed         : {cfg.seed}")
    print("=" * 55)

    trainer = train(cfg)

    # Print final validation metrics.
    metrics = trainer.evaluate()
    print("\nFinal validation metrics:")
    for k, v in sorted(metrics.items()):
        print(f"  {k:30s} : {v}")


if __name__ == "__main__":
    main()