"""
ner_model.py
--------
BiomedBERT wrapped with a token classification head for BC5CDR NER.

Uses HuggingFace AutoModelForTokenClassification so the linear
classification layer (hidden_size -> num_labels) is created automatically.
"""

from __future__ import annotations

from transformers import AutoModelForTokenClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# BC5CDR label scheme (IOB2)
# ---------------------------------------------------------------------------

# Ordered to match the integer tag ids used in the tner/bc5cdr dataset.
LABEL_NAMES = ["O", "B-Chemical", "I-Chemical", "B-Disease", "I-Disease"]

NUM_LABELS = len(LABEL_NAMES)

# Bidirectional mappings between label strings and integer ids.
label2id = {label: idx for idx, label in enumerate(LABEL_NAMES)}
id2label = {idx: label for idx, label in enumerate(LABEL_NAMES)}

# ---------------------------------------------------------------------------
# Pre-trained backbone
# ---------------------------------------------------------------------------

MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_model(
    model_name: str = MODEL_NAME,
    num_labels: int = NUM_LABELS,
) -> AutoModelForTokenClassification:
    """Load BiomedBERT with a token classification head on top.

    The label mappings are stored inside the model config so that
    ``model.config.id2label`` / ``model.config.label2id`` are available
    during inference and when pushing to the Hub.

    Parameters
    ----------
    model_name : str
        HuggingFace model hub identifier for the pre-trained backbone.
    num_labels : int
        Number of IOB2 entity tags (default 5 for BC5CDR).

    Returns
    -------
    AutoModelForTokenClassification
        Ready-to-train model with randomly initialised classification head.
    """
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    return model


def build_tokenizer(model_name: str = MODEL_NAME) -> AutoTokenizer:
    """Load the fast tokenizer that matches the backbone.

    Parameters
    ----------
    model_name : str
        Same identifier used for ``build_model``.

    Returns
    -------
    AutoTokenizer
        A fast (Rust-backed) tokenizer instance.
    """
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Label mappings")
    print(f"  label2id : {label2id}")
    print(f"  id2label : {id2label}")
    print(f"  num_labels : {NUM_LABELS}")

    print(f"\nLoading model: {MODEL_NAME} ...")
    model = build_model()
    print(f"  Parameters : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Config id2label : {model.config.id2label}")
    print(f"  Config label2id : {model.config.label2id}")

    tokenizer = build_tokenizer()
    sample = tokenizer("Aspirin treats headache.", return_tensors="pt")
    outputs = model(**sample)
    print(f"\n  Logits shape : {outputs.logits.shape}")
    print("Done.")