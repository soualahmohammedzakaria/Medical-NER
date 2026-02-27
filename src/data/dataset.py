"""
dataset.py
----------
Load the BC5CDR corpus from HuggingFace, tokenize with BiomedBERT,
align NER labels to subword tokens, and expose PyTorch Dataset objects.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from datasets import DatasetDict, load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast

# ---- constants ----

MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

# BC5CDR ships with IOB tags for two entity types: Chemical and Disease.
# HuggingFace exposes them as integer ids; we keep a readable map for logging.
BC5CDR_LABEL_NAMES: List[str] = [
    "O",
    "B-Chemical",
    "I-Chemical",
    "B-Disease",
    "I-Disease",
]

# Label id assigned to special tokens ([CLS], [SEP], [PAD]) and to non-first
# subword pieces so the loss function ignores them during training.
IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Tokenization + label alignment
# ---------------------------------------------------------------------------

def tokenize_and_align_labels(
    examples: Dict[str, List],
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 512,
) -> Dict[str, List]:
    """Tokenize a batch of pre-split token lists and realign NER labels.

    For each word the tokenizer may produce multiple subword pieces.
    We assign the original label only to the *first* subword piece and
    set all subsequent pieces to IGNORE_INDEX (-100) so CrossEntropyLoss
    skips them automatically.

    Parameters
    ----------
    examples : dict
        A batch from a HuggingFace Dataset with keys "tokens" and "ner_tags".
    tokenizer : PreTrainedTokenizerFast
        The BiomedBERT (or any fast) tokenizer.
    max_length : int
        Maximum sequence length after tokenization (truncation applied).

    Returns
    -------
    dict
        The tokenized batch with an added "labels" key.
    """
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,   # input is already word-level
        truncation=True,
        max_length=max_length,
        padding="max_length",       # pad to max_length for uniform tensors
    )

    all_labels: List[List[int]] = []

    for i, original_labels in enumerate(examples["ner_tags"]):
        # word_ids() maps each subword position to its original word index,
        # returning None for special tokens ([CLS], [SEP], [PAD]).
        word_ids = tokenized.word_ids(batch_index=i)

        label_ids: List[int] = []
        previous_word_id: Optional[int] = None

        for word_id in word_ids:
            if word_id is None:
                # special token - ignore in loss
                label_ids.append(IGNORE_INDEX)
            elif word_id != previous_word_id:
                # first subword of a new word - keep the real label
                label_ids.append(original_labels[word_id])
            else:
                # continuation subword of the same word - ignore in loss
                label_ids.append(IGNORE_INDEX)
            previous_word_id = word_id

        all_labels.append(label_ids)

    tokenized["labels"] = all_labels
    return tokenized


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------

def load_bc5cdr(
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 512,
) -> DatasetDict:
    """Download BC5CDR from HuggingFace and return tokenized splits.

    The dataset is loaded via ``datasets.load_dataset("tner/bc5cdr")``.
    Each split (train / validation / test) is tokenized in-place using
    ``datasets.map`` with batched processing for speed.

    Returns
    -------
    DatasetDict
        Keys: "train", "validation", "test". Each split contains columns
        input_ids, attention_mask, labels (all List[int]).
    """
    raw = load_dataset("tner/bc5cdr")

    tokenized = raw.map(
        lambda batch: tokenize_and_align_labels(batch, tokenizer, max_length),
        batched=True,
        remove_columns=raw["train"].column_names,  # drop raw text columns
    )

    # HuggingFace datasets are Arrow tables; set format to pytorch so
    # __getitem__ returns tensors directly.
    tokenized.set_format("torch")
    return tokenized


def get_tokenizer(model_name: str = MODEL_NAME) -> PreTrainedTokenizerFast:
    """Instantiate the BiomedBERT fast tokenizer."""
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


# ---------------------------------------------------------------------------
# PyTorch Dataset wrapper (optional - for custom DataLoader usage)
# ---------------------------------------------------------------------------

class NERDataset(Dataset):
    """Thin PyTorch Dataset that wraps a single HuggingFace split.

    Use this when you need a standard ``torch.utils.data.DataLoader``
    instead of the HuggingFace Trainer.

    Parameters
    ----------
    hf_dataset
        A single split (e.g. tokenized["train"]) already in "torch" format.
    """

    def __init__(self, hf_dataset) -> None:
        self.data = hf_dataset

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": item["labels"],
        }


# ---------------------------------------------------------------------------
# Convenience entry-point
# ---------------------------------------------------------------------------

def prepare_datasets(
    model_name: str = MODEL_NAME,
    max_length: int = 512,
) -> tuple[DatasetDict, PreTrainedTokenizerFast]:
    """One-call helper: load tokenizer + tokenized BC5CDR splits.

    Returns
    -------
    (datasets, tokenizer)
        datasets is a DatasetDict with train / validation / test splits.
    """
    tokenizer = get_tokenizer(model_name)
    datasets = load_bc5cdr(tokenizer, max_length)
    return datasets, tokenizer


if __name__ == "__main__":
    # Quick smoke test: load one batch and print shapes.
    ds, tok = prepare_datasets(max_length=128)
    sample = ds["train"][0]
    print(f"input_ids   : {sample['input_ids'].shape}")
    print(f"attention   : {sample['attention_mask'].shape}")
    print(f"labels      : {sample['labels'].shape}")
    print(f"Train size  : {len(ds['train'])}")
    print(f"Val size    : {len(ds['validation'])}")
    print(f"Test size   : {len(ds['test'])}")