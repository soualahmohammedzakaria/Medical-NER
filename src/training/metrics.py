"""
metrics.py
----------
Compute entity-level NER metrics (precision, recall, F1) via seqeval.

Designed to plug into HuggingFace Trainer's compute_metrics callback, but
also usable standalone.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from src.models.ner_model import LABEL_NAMES, id2label

# The ignore index used during label alignment (non-first subword tokens,
# special tokens, padding). Must match the value in dataset.py.
IGNORE_INDEX = -100


def decode_predictions(
    preds: np.ndarray,
    labels: np.ndarray,
) -> Tuple[List[List[str]], List[List[str]]]:
    """Convert integer prediction/label arrays to lists of label-name sequences.

    Positions where the gold label equals IGNORE_INDEX are dropped so that
    seqeval only evaluates real (first-subword) tokens.

    Parameters
    ----------
    preds : np.ndarray, shape (batch, seq_len)
        Argmax'd prediction ids.
    labels : np.ndarray, shape (batch, seq_len)
        Gold label ids (may contain -100 for ignored positions).

    Returns
    -------
    (pred_tags, gold_tags)
        Each is a list of sentences, where each sentence is a list of
        IOB2 label strings with ignored positions removed.
    """
    pred_tags: List[List[str]] = []
    gold_tags: List[List[str]] = []

    for pred_seq, gold_seq in zip(preds, labels):
        p_tags: List[str] = []
        g_tags: List[str] = []
        for p, g in zip(pred_seq, gold_seq):
            if g == IGNORE_INDEX:
                continue
            p_tags.append(id2label[int(p)])
            g_tags.append(id2label[int(g)])
        pred_tags.append(p_tags)
        gold_tags.append(g_tags)

    return pred_tags, gold_tags


def compute_metrics(eval_pred) -> Dict[str, float]:
    """HuggingFace Trainer-compatible compute_metrics callback.

    Parameters
    ----------
    eval_pred : transformers.EvalPrediction
        Contains .predictions (logits) and .label_ids.

    Returns
    -------
    dict
        Keys: precision, recall, f1 (all entity-level, micro-averaged).
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    pred_tags, gold_tags = decode_predictions(preds, labels)

    return {
        "precision": precision_score(gold_tags, pred_tags),
        "recall": recall_score(gold_tags, pred_tags),
        "f1": f1_score(gold_tags, pred_tags),
    }


def full_classification_report(
    preds: np.ndarray,
    labels: np.ndarray,
) -> str:
    """Return the seqeval per-entity-type classification report as a string."""
    pred_tags, gold_tags = decode_predictions(preds, labels)
    return classification_report(gold_tags, pred_tags)