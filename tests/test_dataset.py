"""
tests/test_dataset.py

Unit tests for the BC5CDR dataset loading and tokenization pipeline.

Covers:
 1. Tokenization output has correct shape (input_ids, attention_mask, labels).
 2. Label alignment is correct: -100 appears at non-first subword positions
    and at special tokens ([CLS], [SEP], [PAD]).
 3. No label leakage between train and test splits (no identical samples).
"""

from __future__ import annotations

import pytest

from src.data.dataset import (
    BC5CDR_LABEL_NAMES,
    IGNORE_INDEX,
    _load_bc5cdr_raw,
    get_tokenizer,
    tokenize_and_align_labels,
    load_bc5cdr,
    NERDataset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tokenizer():
    """Load the BiomedBERT tokenizer once for the whole module."""
    return get_tokenizer()


@pytest.fixture(scope="module")
def raw_splits():
    """Download the raw (un-tokenized) BC5CDR splits once."""
    return _load_bc5cdr_raw()


@pytest.fixture(scope="module")
def tokenized_splits(tokenizer):
    """Return tokenized BC5CDR splits (max_length=128 for speed)."""
    return load_bc5cdr(tokenizer, max_length=128)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_LABEL_IDS = set(range(len(BC5CDR_LABEL_NAMES)))  # {0,1,2,3,4}


def _as_list(tensor_or_list):
    """Convert a torch.Tensor or plain list to a Python list."""
    if hasattr(tensor_or_list, "tolist"):
        return tensor_or_list.tolist()
    return list(tensor_or_list)


# ===========================================================================
# 1. Tokenization output shape
# ===========================================================================

class TestTokenizationShape:
    """Verify that tokenized samples have the right keys and dimensions."""

    def test_all_splits_present(self, tokenized_splits):
        assert "train" in tokenized_splits
        assert "validation" in tokenized_splits
        assert "test" in tokenized_splits

    def test_required_columns(self, tokenized_splits):
        for split_name in ("train", "validation", "test"):
            cols = tokenized_splits[split_name].column_names
            assert "input_ids" in cols
            assert "attention_mask" in cols
            assert "labels" in cols

    @pytest.mark.parametrize("split", ["train", "validation", "test"])
    def test_sequence_length_matches(self, tokenized_splits, split):
        """input_ids, attention_mask, and labels must all be length 128."""
        sample = tokenized_splits[split][0]
        ids = _as_list(sample["input_ids"])
        mask = _as_list(sample["attention_mask"])
        labels = _as_list(sample["labels"])

        assert len(ids) == 128
        assert len(mask) == 128
        assert len(labels) == 128

    @pytest.mark.parametrize("split", ["train", "validation", "test"])
    def test_lengths_consistent_across_keys(self, tokenized_splits, split):
        """All three tensors in every sample must have identical length."""
        for idx in (0, 1, min(5, len(tokenized_splits[split]) - 1)):
            sample = tokenized_splits[split][idx]
            n = len(_as_list(sample["input_ids"]))
            assert len(_as_list(sample["attention_mask"])) == n
            assert len(_as_list(sample["labels"])) == n

    def test_non_empty_splits(self, tokenized_splits):
        assert len(tokenized_splits["train"]) > 0
        assert len(tokenized_splits["validation"]) > 0
        assert len(tokenized_splits["test"]) > 0


# ===========================================================================
# 2. Label alignment
# ===========================================================================

class TestLabelAlignment:
    """Verify that -100 appears only where expected and real labels are valid."""

    def test_cls_sep_tokens_have_ignore_index(self, tokenized_splits, tokenizer):
        """[CLS] (pos 0) and first [SEP] must have label == -100."""
        sample = tokenized_splits["train"][0]
        ids = _as_list(sample["input_ids"])
        labels = _as_list(sample["labels"])

        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id

        # Position 0 should be [CLS]
        assert ids[0] == cls_id
        assert labels[0] == IGNORE_INDEX

        # Find first [SEP]
        sep_pos = ids.index(sep_id)
        assert labels[sep_pos] == IGNORE_INDEX

    def test_pad_tokens_have_ignore_index(self, tokenized_splits, tokenizer):
        """Every [PAD] position must have label == -100."""
        sample = tokenized_splits["train"][0]
        ids = _as_list(sample["input_ids"])
        labels = _as_list(sample["labels"])
        pad_id = tokenizer.pad_token_id

        for pos, token_id in enumerate(ids):
            if token_id == pad_id:
                assert labels[pos] == IGNORE_INDEX, (
                    f"PAD at position {pos} has label {labels[pos]}, "
                    f"expected {IGNORE_INDEX}"
                )

    def test_non_first_subword_has_ignore_index(self, tokenizer, raw_splits):
        """When a word splits into >1 subwords, only the first keeps the label.

        We take a raw example, tokenize it manually, and verify that
        continuation subword positions are set to -100.
        """
        # Pick the first raw training example
        example = raw_splits["train"][0]
        tokens = example["tokens"]
        ner_tags = example["ner_tags"]

        # Build a single-example batch (the function expects batched input)
        batch = {"tokens": [tokens], "ner_tags": [ner_tags]}
        result = tokenize_and_align_labels(batch, tokenizer, max_length=128)

        word_ids = result.word_ids(batch_index=0)
        labels = result["labels"][0]

        prev_word_id = None
        for pos, wid in enumerate(word_ids):
            if wid is None:
                # special token
                assert labels[pos] == IGNORE_INDEX
            elif wid == prev_word_id:
                # continuation subword -- must be -100
                assert labels[pos] == IGNORE_INDEX, (
                    f"Position {pos} is a continuation subword (word_id={wid}) "
                    f"but has label {labels[pos]} instead of {IGNORE_INDEX}"
                )
            else:
                # first subword -- must carry a real label
                assert labels[pos] in VALID_LABEL_IDS, (
                    f"Position {pos} is the first subword of word {wid} "
                    f"but has unexpected label {labels[pos]}"
                )
            prev_word_id = wid

    def test_real_labels_are_valid(self, tokenized_splits):
        """Every non-ignored label must be in {0, 1, 2, 3, 4}."""
        for idx in range(min(50, len(tokenized_splits["train"]))):
            labels = _as_list(tokenized_splits["train"][idx]["labels"])
            for pos, lbl in enumerate(labels):
                if lbl != IGNORE_INDEX:
                    assert lbl in VALID_LABEL_IDS, (
                        f"Sample {idx}, pos {pos}: invalid label {lbl}"
                    )

    def test_at_least_one_real_label_per_sample(self, tokenized_splits):
        """Every sample should have at least one non-ignored position."""
        for idx in range(min(50, len(tokenized_splits["train"]))):
            labels = _as_list(tokenized_splits["train"][idx]["labels"])
            real = [l for l in labels if l != IGNORE_INDEX]
            assert len(real) > 0, f"Sample {idx} has no real labels at all"


# ===========================================================================
# 3. No label leakage between train and test
# ===========================================================================

class TestNoLeakage:
    """Ensure train and test splits do not share identical samples."""

    @staticmethod
    def _sentence_key(example) -> str:
        """Create a hashable string from the raw token list."""
        return " ".join(example["tokens"])

    def test_no_identical_sentences_in_train_and_test(self, raw_splits):
        """No sentence should appear in both train and test."""
        train_sentences = {
            self._sentence_key(raw_splits["train"][i])
            for i in range(len(raw_splits["train"]))
        }
        test_sentences = {
            self._sentence_key(raw_splits["test"][i])
            for i in range(len(raw_splits["test"]))
        }
        overlap = train_sentences & test_sentences
        assert len(overlap) == 0, (
            f"{len(overlap)} sentence(s) appear in both train and test. "
            f"Examples: {list(overlap)[:3]}"
        )

    def test_no_identical_sentences_in_train_and_validation(self, raw_splits):
        """No sentence should appear in both train and validation."""
        train_sentences = {
            self._sentence_key(raw_splits["train"][i])
            for i in range(len(raw_splits["train"]))
        }
        val_sentences = {
            self._sentence_key(raw_splits["validation"][i])
            for i in range(len(raw_splits["validation"]))
        }
        overlap = train_sentences & val_sentences
        assert len(overlap) == 0, (
            f"{len(overlap)} sentence(s) appear in both train and validation. "
            f"Examples: {list(overlap)[:3]}"
        )

    def test_no_identical_tokenized_ids_in_train_and_test(self, tokenized_splits):
        """Cross-check at the tokenized level too (input_ids tuples)."""
        train_ids = {
            tuple(_as_list(tokenized_splits["train"][i]["input_ids"]))
            for i in range(len(tokenized_splits["train"]))
        }
        test_ids = {
            tuple(_as_list(tokenized_splits["test"][i]["input_ids"]))
            for i in range(len(tokenized_splits["test"]))
        }
        overlap = train_ids & test_ids
        assert len(overlap) == 0, (
            f"{len(overlap)} tokenized sequence(s) appear in both train and test."
        )


# ===========================================================================
# Bonus: NERDataset wrapper
# ===========================================================================

class TestNERDataset:
    """Verify the PyTorch Dataset wrapper."""

    def test_len(self, tokenized_splits):
        ds = NERDataset(tokenized_splits["train"])
        assert len(ds) == len(tokenized_splits["train"])

    def test_getitem_keys(self, tokenized_splits):
        ds = NERDataset(tokenized_splits["train"])
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item

    def test_getitem_shapes_match(self, tokenized_splits):
        ds = NERDataset(tokenized_splits["train"])
        item = ds[0]
        n = item["input_ids"].shape[0]
        assert item["attention_mask"].shape[0] == n
        assert item["labels"].shape[0] == n
