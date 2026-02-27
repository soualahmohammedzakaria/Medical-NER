"""
download.py
-----------
Download and cache BC5CDR locally, print dataset statistics, and save
a handful of tokenized examples to JSON for manual inspection.

Usage:
    python -m src.data.download
    python -m src.data.download --max-length 256 --num-examples 10
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import List

from datasets import DatasetDict, load_dataset

from src.data.dataset import (
    BC5CDR_LABEL_NAMES,
    MODEL_NAME,
    _load_bc5cdr_raw,
    get_tokenizer,
    load_bc5cdr,
)

# Where to dump the example JSON file
DEFAULT_OUTPUT_DIR = Path("data/processed")
DEFAULT_EXAMPLES_FILE = "example_tokenized_samples.json"


# -------------------------------------------------------------------
# Statistics helpers
# -------------------------------------------------------------------

def count_entities_per_type(
    tags_column: List[List[int]],
    label_names: List[str],
) -> Counter:
    """Count how many entity *spans* of each type appear in a split.

    A span starts whenever we see a B-* tag.  We attribute the span to
    the entity type embedded in that tag name (e.g. "B-Disease" -> "Disease").
    """
    entity_counts: Counter = Counter()
    for tag_seq in tags_column:
        for tag_id in tag_seq:
            tag_name = label_names[tag_id]
            if tag_name.startswith("B-"):
                # extract the entity type after the "B-" prefix
                entity_type = tag_name[2:]
                entity_counts[entity_type] += 1
    return entity_counts


def average_sequence_length(tokens_column: List[List[str]]) -> float:
    """Return the mean number of *word-level* tokens per sample."""
    total = sum(len(seq) for seq in tokens_column)
    return total / len(tokens_column) if tokens_column else 0.0


def print_split_stats(
    split_name: str,
    tokens_column: List[List[str]],
    tags_column: List[List[int]],
    label_names: List[str],
) -> None:
    """Print a summary block for one dataset split."""
    n_samples = len(tokens_column)
    avg_len = average_sequence_length(tokens_column)
    entity_counts = count_entities_per_type(tags_column, label_names)

    print(f"\n  [{split_name}]")
    print(f"    Samples           : {n_samples}")
    print(f"    Avg token length  : {avg_len:.1f}")
    print(f"    Entity spans      :")
    for etype, count in sorted(entity_counts.items()):
        print(f"      {etype:12s} : {count}")


# -------------------------------------------------------------------
# Example export
# -------------------------------------------------------------------

def save_tokenized_examples(
    tokenized_ds: DatasetDict,
    label_names: List[str],
    num_examples: int = 5,
    output_path: Path | None = None,
) -> Path:
    """Save a few tokenized train samples to a JSON file for inspection.

    Each saved example includes the raw token ids, attention mask, aligned
    labels, and a human-readable decoded-tokens / label-names view.
    """
    if output_path is None:
        output_path = DEFAULT_OUTPUT_DIR / DEFAULT_EXAMPLES_FILE

    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = get_tokenizer()
    examples = []

    for idx in range(min(num_examples, len(tokenized_ds["train"]))):
        item = tokenized_ds["train"][idx]

        # .tolist() converts tensors back to plain Python lists for JSON
        input_ids = item["input_ids"].tolist()
        attention_mask = item["attention_mask"].tolist()
        labels = item["labels"].tolist()

        # decode each token id individually so we can inspect subwords
        decoded_tokens = tokenizer.convert_ids_to_tokens(input_ids)

        # map label ids to readable names; -100 becomes "IGN" (ignored)
        readable_labels = [
            label_names[lid] if lid != -100 else "IGN"
            for lid in labels
        ]

        # only keep non-padding positions for a cleaner view
        seq_len = sum(attention_mask)
        examples.append({
            "index": idx,
            "input_ids": input_ids[:seq_len],
            "attention_mask": attention_mask[:seq_len],
            "labels": labels[:seq_len],
            "decoded_tokens": decoded_tokens[:seq_len],
            "readable_labels": readable_labels[:seq_len],
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    return output_path


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main(max_length: int = 512, num_examples: int = 5) -> None:
    """Download BC5CDR, print stats, tokenize, and export examples."""

    print("=" * 55)
    print("  BC5CDR -- Download & Statistics")
    print("=" * 55)

    # -- 1. Download raw dataset (HuggingFace caches it automatically) --
    print("\nDownloading / loading BC5CDR from HuggingFace hub ...")
    raw = _load_bc5cdr_raw()

    # -- 2. Print per-split statistics on the raw (word-level) data --
    print("\n--- Raw dataset statistics ---")
    for split_name in ("train", "validation", "test"):
        if split_name not in raw:
            continue
        split = raw[split_name]
        print_split_stats(
            split_name,
            split["tokens"],
            split["ner_tags"],
            BC5CDR_LABEL_NAMES,
        )

    # -- 3. Tokenize with BiomedBERT --
    print(f"\nTokenizing with {MODEL_NAME} (max_length={max_length}) ...")
    tokenizer = get_tokenizer()
    tokenized = load_bc5cdr(tokenizer, max_length=max_length)

    print(f"  Tokenized train columns : {tokenized['train'].column_names}")
    print(f"  Tensor format           : {tokenized['train'].format['type']}")

    # -- 4. Save example tokenized samples to JSON --
    out_path = save_tokenized_examples(
        tokenized,
        BC5CDR_LABEL_NAMES,
        num_examples=num_examples,
    )
    print(f"\nSaved {num_examples} tokenized examples -> {out_path}")
    print("Done.\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download BC5CDR & print stats")
    parser.add_argument(
        "--max-length", type=int, default=512,
        help="Max subword sequence length for tokenization",
    )
    parser.add_argument(
        "--num-examples", type=int, default=5,
        help="Number of tokenized examples to save to JSON",
    )
    args = parser.parse_args()
    main(max_length=args.max_length, num_examples=args.num_examples)