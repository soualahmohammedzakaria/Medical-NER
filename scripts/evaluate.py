"""
evaluate.py

CLI entry-point for evaluating a trained Medical-NER checkpoint.

Usage:
    python scripts/evaluate.py --checkpoint outputs/models/best
    python scripts/evaluate.py --checkpoint outputs/models/best --split validation
    python scripts/evaluate.py --checkpoint outputs/models/best --output results.json
"""

from __future__ import annotations

import argparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained NER model on BC5CDR",
    )
    p.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to saved model directory (e.g. outputs/models/best)",
    )
    p.add_argument(
        "--split", type=str, default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate on (default: test)",
    )
    p.add_argument(
        "--batch-size", type=int, default=32,
        help="Inference batch size (default: 32)",
    )
    p.add_argument(
        "--max-length", type=int, default=512,
        help="Max subword sequence length (default: 512)",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Path to save JSON results (default: outputs/results/eval_<split>.json)",
    )
    p.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run inference on (default: auto)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Deferred import so --help stays fast.
    from src.evaluation.evaluator import evaluate

    evaluate(
        checkpoint_dir=args.checkpoint,
        split=args.split,
        batch_size=args.batch_size,
        max_length=args.max_length,
        output_path=args.output,
        seed=args.seed,
        device_preference=args.device,
    )


if __name__ == "__main__":
    main()
