"""
evaluator.py

Load a saved NER checkpoint, run inference on the test split, and compute
entity-level precision / recall / F1 using seqeval with a per-entity-type
breakdown (Chemical vs Disease). Results are printed and saved to JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.data.dataset import load_bc5cdr
from src.models.ner_model import LABEL_NAMES, id2label
from src.training.metrics import IGNORE_INDEX, decode_predictions
from src.utils.helpers import get_device, seed_everything


# ---------------------------------------------------------------------------
# Collect predictions
# ---------------------------------------------------------------------------

def collect_predictions(
    model: AutoModelForTokenClassification,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the model over a DataLoader and collect all predictions and labels.

    Returns
    -------
    (all_preds, all_labels)
        Both arrays have shape (total_samples, seq_len) with integer ids.
    """
    model.eval()
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]  # keep on CPU for collection

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu().numpy()
            preds = np.argmax(logits, axis=-1)

            all_preds.append(preds)
            all_labels.append(labels.numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Per-entity-type metrics
# ---------------------------------------------------------------------------

def per_entity_metrics(
    pred_tags: List[List[str]],
    gold_tags: List[List[str]],
) -> Dict[str, Dict[str, float]]:
    """Compute precision, recall, F1 for each entity type individually.

    seqeval's classification_report with output_dict=True gives per-type
    rows keyed by entity name (e.g. "Chemical", "Disease").

    Returns
    -------
    dict
        Nested dict: {entity_type: {precision, recall, f1, support}}.
    """
    report: Dict[str, Any] = classification_report(
        gold_tags, pred_tags, output_dict=True,
    )

    # seqeval returns keys like "Chemical", "Disease", plus aggregates
    # ("micro avg", "macro avg", "weighted avg").
    entity_types = [k for k in report if k not in (
        "micro avg", "macro avg", "weighted avg",
    )]

    per_type: Dict[str, Dict[str, float]] = {}
    for etype in entity_types:
        row = report[etype]
        per_type[etype] = {
            "precision": round(row["precision"], 4),
            "recall": round(row["recall"], 4),
            "f1": round(row["f1-score"], 4),
            "support": int(row["support"]),
        }
    return per_type


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate(
    checkpoint_dir: str,
    split: str = "test",
    batch_size: int = 32,
    max_length: int = 512,
    output_path: str | None = None,
    seed: int = 42,
    device_preference: str = "auto",
) -> Dict[str, Any]:
    """End-to-end evaluation of a saved checkpoint.

    Parameters
    ----------
    checkpoint_dir : str
        Path to a directory containing a saved model and tokenizer
        (e.g. outputs/models/best).
    split : str
        Which dataset split to evaluate on ("test", "validation", "train").
    batch_size : int
        Inference batch size.
    max_length : int
        Max subword sequence length (must match training).
    output_path : str or None
        If provided, save the results dict to this JSON file.
    seed : int
        Random seed for reproducibility.
    device_preference : str
        "auto", "cpu", "cuda", or "mps".

    Returns
    -------
    dict
        Complete results including overall and per-entity metrics.
    """
    seed_everything(seed)
    device = get_device(device_preference)
    print(f"Device: {device}")

    # load model and tokenizer from checkpoint
    print(f"Loading checkpoint from {checkpoint_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(checkpoint_dir)
    model.to(device)

    # load and tokenize the dataset
    print(f"Loading BC5CDR ({split} split) ...")
    datasets = load_bc5cdr(tokenizer, max_length=max_length)
    eval_dataset = datasets[split]

    dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # collect predictions
    print("Running inference ...")
    all_preds, all_labels = collect_predictions(model, dataloader, device)

    # decode to IOB2 tag strings
    pred_tags, gold_tags = decode_predictions(all_preds, all_labels)

    # overall (micro-averaged) metrics
    overall = {
        "precision": round(precision_score(gold_tags, pred_tags), 4),
        "recall": round(recall_score(gold_tags, pred_tags), 4),
        "f1": round(f1_score(gold_tags, pred_tags), 4),
    }

    # per-entity-type breakdown
    per_type = per_entity_metrics(pred_tags, gold_tags)

    # full text report
    report_str = classification_report(gold_tags, pred_tags)

    # assemble results
    results: Dict[str, Any] = {
        "checkpoint": str(checkpoint_dir),
        "split": split,
        "num_samples": len(pred_tags),
        "overall": overall,
        "per_entity": per_type,
    }

    # print
    print("\n" + "=" * 55)
    print("  Evaluation Results")
    print("=" * 55)
    print(f"  Split     : {split}")
    print(f"  Samples   : {len(pred_tags)}")
    print(f"  Precision : {overall['precision']}")
    print(f"  Recall    : {overall['recall']}")
    print(f"  F1        : {overall['f1']}")
    print("\nPer-entity breakdown:")
    print(report_str)

    # save to JSON
    if output_path is None:
        out_dir = Path("outputs/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"eval_{split}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_path}")

    return results
