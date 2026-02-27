"""
analyze_errors.py

CLI script that loads a saved checkpoint, runs inference on a BC5CDR
split, and performs a full error analysis (false positives, false
negatives, boundary errors, negation errors).

Usage:
    python scripts/analyze_errors.py --checkpoint outputs/models/best
    python scripts/analyze_errors.py --checkpoint outputs/models/best --split validation --top-n 20
"""

from __future__ import annotations

import argparse

import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.data.dataset import _load_bc5cdr_raw, load_bc5cdr
from src.evaluation.error_analysis import run_error_analysis
from src.evaluation.evaluator import collect_predictions
from src.models.ner_model import id2label
from src.training.metrics import IGNORE_INDEX
from src.utils.helpers import get_device, seed_everything


def _recover_aligned_tokens(
    raw_tokens: list[list[str]],
    labels_array: np.ndarray,
) -> list[list[str]]:
    """Build a per-sentence token list aligned with decoded predictions.

    During tokenization, non-first subwords and special tokens receive
    label = -100. decode_predictions strips those positions, so the
    resulting tag list is shorter than the tokenized sequence. We need
    a matching token list of the same length.

    Strategy: for every sentence, walk the label array and keep only the
    positions where label != -100. Those positions correspond 1-to-1 to
    the original word-level tokens (one per word, since only the first
    subword gets a real label). If the tokenized sentence was truncated,
    we slice the raw tokens to match.
    """
    aligned: list[list[str]] = []
    for sent_idx, label_seq in enumerate(labels_array):
        # count how many real (non-ignored) tokens survived
        n_real = int(np.sum(label_seq != IGNORE_INDEX))
        # take that many tokens from the raw word list
        toks = raw_tokens[sent_idx][:n_real]
        # pad with "<UNK>" if raw tokens are shorter (shouldn't happen normally)
        while len(toks) < n_real:
            toks.append("<UNK>")
        aligned.append(toks)
    return aligned


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NER error analysis on BC5CDR")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to saved model directory")
    p.add_argument("--split", type=str, default="test",
                   choices=["train", "validation", "test"],
                   help="Dataset split (default: test)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--top-n", type=int, default=10,
                   help="Number of items for FP/FN/boundary/negation lists")
    p.add_argument("--output", type=str, default=None,
                   help="JSON output path (default: outputs/results/error_analysis.json)")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)
    device = get_device(args.device)

    print(f"Device: {device}")
    print(f"Loading checkpoint from {args.checkpoint} ...")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.checkpoint)
    model.to(device)

    # tokenized dataset (for model inference)
    print(f"Loading BC5CDR ({args.split} split) ...")
    tokenized = load_bc5cdr(tokenizer, max_length=args.max_length)
    eval_dataset = tokenized[args.split]
    dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    # raw dataset (for original word-level tokens)
    raw = _load_bc5cdr_raw()
    raw_tokens: list[list[str]] = raw[args.split]["tokens"]

    # run inference
    print("Running inference ...")
    all_preds, all_labels = collect_predictions(model, dataloader, device)

    # decode to IOB2 strings (only non-ignored positions)
    from src.training.metrics import decode_predictions
    pred_tags, gold_tags = decode_predictions(all_preds, all_labels)

    # recover word-level tokens aligned with the decoded tag sequences
    all_tokens = _recover_aligned_tokens(raw_tokens, all_labels)

    # run error analysis
    run_error_analysis(
        pred_tags=pred_tags,
        gold_tags=gold_tags,
        all_tokens=all_tokens,
        top_n=args.top_n,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()