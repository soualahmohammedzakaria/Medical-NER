"""
tests/test_model.py

Unit tests for the NER model architecture and label mappings.

Covers:
 - Label constants are consistent (label2id / id2label round-trip, count).
 - build_model returns correct architecture with right num_labels.
 - Forward pass produces logits with expected shape on dummy input.
 - Model config stores id2label / label2id correctly.
 - Model save / load round-trip preserves config and output shape.
 - build_tokenizer returns a fast tokenizer.
"""

from __future__ import annotations

import pytest
import tempfile
import torch
from pathlib import Path

from src.models.ner_model import (
    LABEL_NAMES,
    NUM_LABELS,
    MODEL_NAME,
    label2id,
    id2label,
    build_model,
    build_tokenizer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    """Build the BiomedBERT NER model once for the whole module."""
    return build_model()


@pytest.fixture(scope="module")
def tokenizer():
    """Load the BiomedBERT tokenizer once for the whole module."""
    return build_tokenizer()


# ===========================================================================
# 1. Label constants
# ===========================================================================

class TestLabelConstants:
    """Verify label mappings are internally consistent."""

    def test_num_labels_matches_label_names(self):
        assert NUM_LABELS == len(LABEL_NAMES)

    def test_label2id_length(self):
        assert len(label2id) == NUM_LABELS

    def test_id2label_length(self):
        assert len(id2label) == NUM_LABELS

    def test_label2id_round_trip(self):
        """label -> id -> label should be identity."""
        for label, idx in label2id.items():
            assert id2label[idx] == label

    def test_id2label_round_trip(self):
        """id -> label -> id should be identity."""
        for idx, label in id2label.items():
            assert label2id[label] == idx

    def test_o_tag_is_zero(self):
        assert label2id["O"] == 0

    def test_expected_labels_present(self):
        expected = {"O", "B-Chemical", "I-Chemical", "B-Disease", "I-Disease"}
        assert set(LABEL_NAMES) == expected


# ===========================================================================
# 2. Model architecture
# ===========================================================================

class TestBuildModel:
    """Verify build_model produces a properly configured model."""

    def test_returns_correct_type(self, model):
        from transformers import AutoModelForTokenClassification
        assert isinstance(model, AutoModelForTokenClassification.__class__) or \
               hasattr(model, "classifier")

    def test_num_labels_in_config(self, model):
        assert model.config.num_labels == NUM_LABELS

    def test_config_id2label(self, model):
        """Model config should store our id2label mapping."""
        cfg_id2label = model.config.id2label
        for idx, label in id2label.items():
            assert cfg_id2label[idx] == label

    def test_config_label2id(self, model):
        """Model config should store our label2id mapping."""
        cfg_label2id = model.config.label2id
        for label, idx in label2id.items():
            assert cfg_label2id[label] == idx

    def test_has_classifier_layer(self, model):
        """The model should have a classifier head."""
        assert hasattr(model, "classifier")

    def test_classifier_output_dim(self, model):
        """Classifier output dimension must equal NUM_LABELS."""
        out_features = model.classifier.out_features
        assert out_features == NUM_LABELS


# ===========================================================================
# 3. Forward pass
# ===========================================================================

class TestForwardPass:
    """Verify the forward pass produces correct output shapes."""

    def test_logits_shape_single(self, model, tokenizer):
        """Logits should be (1, seq_len, NUM_LABELS) for a single input."""
        inputs = tokenizer("Aspirin treats headache.", return_tensors="pt")
        seq_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model(**inputs)

        assert outputs.logits.shape == (1, seq_len, NUM_LABELS)

    def test_logits_shape_batch(self, model, tokenizer):
        """Logits batch dim should match the input batch size."""
        texts = ["Aspirin treats headache.", "Metformin for diabetes."]
        inputs = tokenizer(texts, return_tensors="pt", padding=True)
        batch_size = inputs["input_ids"].shape[0]
        seq_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model(**inputs)

        assert outputs.logits.shape == (batch_size, seq_len, NUM_LABELS)

    def test_logits_finite(self, model, tokenizer):
        """No NaN or Inf values in logits."""
        inputs = tokenizer("Ibuprofen may cause stomach ulcers.", return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        assert torch.isfinite(logits).all()

    def test_loss_returned_with_labels(self, model, tokenizer):
        """When labels are provided, the output should include a loss."""
        inputs = tokenizer("Aspirin treats headache.", return_tensors="pt")
        seq_len = inputs["input_ids"].shape[1]
        # All-zeros labels (all "O")
        labels = torch.zeros(1, seq_len, dtype=torch.long)

        with torch.no_grad():
            outputs = model(**inputs, labels=labels)

        assert outputs.loss is not None
        assert outputs.loss.ndim == 0  # scalar
        assert torch.isfinite(outputs.loss)


# ===========================================================================
# 4. Save / load round-trip
# ===========================================================================

class TestSaveLoad:
    """Verify model can be saved and reloaded without losing config."""

    def test_round_trip(self, model, tokenizer):
        from transformers import AutoModelForTokenClassification

        with tempfile.TemporaryDirectory() as tmp:
            save_dir = Path(tmp) / "test_checkpoint"

            # Save
            model.save_pretrained(str(save_dir))
            tokenizer.save_pretrained(str(save_dir))

            # Reload
            loaded = AutoModelForTokenClassification.from_pretrained(str(save_dir))

        # Config preserved
        assert loaded.config.num_labels == NUM_LABELS
        for idx, label in id2label.items():
            assert loaded.config.id2label[idx] == label

        # Same output shape on dummy input
        inputs = tokenizer("Test sentence.", return_tensors="pt")
        with torch.no_grad():
            orig_shape = model(**inputs).logits.shape
            loaded_shape = loaded(**inputs).logits.shape
        assert orig_shape == loaded_shape


# ===========================================================================
# 5. Tokenizer
# ===========================================================================

class TestBuildTokenizer:
    """Verify build_tokenizer returns a usable fast tokenizer."""

    def test_is_fast(self, tokenizer):
        assert tokenizer.is_fast

    def test_has_special_tokens(self, tokenizer):
        assert tokenizer.cls_token is not None
        assert tokenizer.sep_token is not None
        assert tokenizer.pad_token is not None

    def test_encode_decode_identity(self, tokenizer):
        text = "Metformin"
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids)
        assert decoded.strip().lower() == text.lower()