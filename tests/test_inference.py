"""
tests/test_inference.py

Unit tests for the inference pipeline (NERPredictor and Entity).

Covers:
 - Entity dataclass and to_dict().
 - _decode_entities correctly merges B-/I- spans.
 - _decode_entities handles edge cases (all-O, entity at start/end).
 - NERPredictor.predict returns Entity objects with valid offsets.
 - predict_batch returns one result list per input text.
 - Batch vs single-sample consistency.
 - Character offsets slice back to the correct surface text.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

import pytest
import torch

from src.inference.predict import Entity, NERPredictor
from src.models.ner_model import (
    NUM_LABELS,
    build_model,
    build_tokenizer,
    id2label,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def checkpoint_dir() -> str:
    """Save a fresh (untrained) model to a temp dir so NERPredictor can load it."""
    tmp = tempfile.mkdtemp(prefix="ner_test_ckpt_")
    model = build_model()
    tokenizer = build_tokenizer()
    model.save_pretrained(tmp)
    tokenizer.save_pretrained(tmp)
    return tmp


@pytest.fixture(scope="module")
def predictor(checkpoint_dir) -> NERPredictor:
    """Build a predictor from the temporary checkpoint."""
    return NERPredictor(checkpoint_dir=checkpoint_dir, device="cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_offsets(pairs: list[tuple[int, int]]) -> torch.Tensor:
    """Build an offset_mapping tensor from a list of (start, end) pairs."""
    return torch.tensor(pairs, dtype=torch.long)


# ===========================================================================
# 1. Entity dataclass
# ===========================================================================

class TestEntity:
    """Tests for the Entity dataclass."""

    def test_fields(self):
        e = Entity(text="Aspirin", label="Chemical", start=0, end=7)
        assert e.text == "Aspirin"
        assert e.label == "Chemical"
        assert e.start == 0
        assert e.end == 7

    def test_to_dict(self):
        e = Entity(text="diabetes", label="Disease", start=10, end=18)
        d = e.to_dict()
        assert isinstance(d, dict)
        assert d == {"text": "diabetes", "label": "Disease",
                      "start": 10, "end": 18}


# ===========================================================================
# 2. _decode_entities logic (unit-level, no model needed)
# ===========================================================================

class TestDecodeEntities:
    """Test the static _decode_entities method directly with synthetic data."""

    def test_single_b_tag(self):
        """A lone B-Chemical with no following I- should produce one entity."""
        text = "Aspirin is good"
        #           [CLS]   Asp    ##rin  is     good   [SEP]
        pred_ids = [0,      1,     2,     0,     0,     0]
        offsets = _make_offsets([
            (0, 0), (0, 3), (3, 7), (8, 10), (11, 15), (0, 0),
        ])
        entities = NERPredictor._decode_entities(text, pred_ids, offsets)
        assert len(entities) == 1
        assert entities[0].text == "Aspirin"
        assert entities[0].label == "Chemical"
        assert entities[0].start == 0
        assert entities[0].end == 7

    def test_b_plus_i_merge(self):
        """B-Disease followed by I-Disease should merge into one entity."""
        text = "heart disease is common"
        #           [CLS]   heart   disease  is     common  [SEP]
        pred_ids = [0,      3,      4,       0,     0,      0]
        offsets = _make_offsets([
            (0, 0), (0, 5), (6, 13), (14, 16), (17, 23), (0, 0),
        ])
        entities = NERPredictor._decode_entities(text, pred_ids, offsets)
        assert len(entities) == 1
        assert entities[0].text == "heart disease"
        assert entities[0].label == "Disease"

    def test_all_o_tags(self):
        """All O tags means no entities."""
        text = "Nothing here"
        pred_ids = [0, 0, 0, 0]
        offsets = _make_offsets([(0, 0), (0, 7), (8, 12), (0, 0)])
        entities = NERPredictor._decode_entities(text, pred_ids, offsets)
        assert entities == []

    def test_multiple_entities(self):
        """Two separate entities in one sentence."""
        text = "Aspirin for headache"
        #           [CLS]   Aspirin  for    headache  [SEP]
        pred_ids = [0,      1,       0,     3,        0]
        offsets = _make_offsets([
            (0, 0), (0, 7), (8, 11), (12, 20), (0, 0),
        ])
        entities = NERPredictor._decode_entities(text, pred_ids, offsets)
        assert len(entities) == 2
        assert entities[0].label == "Chemical"
        assert entities[1].label == "Disease"

    def test_entity_at_end(self):
        """Entity right before [SEP] should still be captured."""
        text = "treats diabetes"
        #           [CLS]   treats  diabetes  [SEP]
        pred_ids = [0,      0,      3,        0]
        offsets = _make_offsets([
            (0, 0), (0, 6), (7, 15), (0, 0),
        ])
        entities = NERPredictor._decode_entities(text, pred_ids, offsets)
        assert len(entities) == 1
        assert entities[0].text == "diabetes"

    def test_special_token_offsets_skipped(self):
        """Entities with (0, 0) offsets (special tokens) should be ignored."""
        text = "test"
        # Pretend CLS itself got a B-Chemical tag -- should be skipped
        pred_ids = [1, 0, 0]
        offsets = _make_offsets([(0, 0), (0, 4), (0, 0)])
        entities = NERPredictor._decode_entities(text, pred_ids, offsets)
        assert entities == []

    def test_adjacent_different_entities(self):
        """B-Chemical immediately followed by B-Disease should give two entities."""
        text = "AspirinDiabetes"
        #           [CLS]   Aspirin   Diabetes  [SEP]
        pred_ids = [0,      1,        3,        0]
        offsets = _make_offsets([
            (0, 0), (0, 7), (7, 15), (0, 0),
        ])
        entities = NERPredictor._decode_entities(text, pred_ids, offsets)
        assert len(entities) == 2
        assert entities[0].label == "Chemical"
        assert entities[1].label == "Disease"


# ===========================================================================
# 3. NERPredictor.predict (end-to-end with real model)
# ===========================================================================

class TestNERPredictorPredict:
    """End-to-end tests using an untrained checkpoint.

    Since the model is untrained, we can NOT assert specific entities are
    found.  We CAN verify the return types, offset validity, and shape.
    """

    def test_returns_list(self, predictor):
        result = predictor.predict("Aspirin treats headache.")
        assert isinstance(result, list)

    def test_entity_types(self, predictor):
        """Every returned element should be an Entity."""
        entities = predictor.predict("Metformin for type 2 diabetes.")
        for e in entities:
            assert isinstance(e, Entity)

    def test_labels_are_known(self, predictor):
        """Entity labels must be Chemical or Disease (the BC5CDR types)."""
        entities = predictor.predict("Ibuprofen may cause ulcers and nausea.")
        for e in entities:
            assert e.label in ("Chemical", "Disease"), (
                f"Unexpected label: {e.label}"
            )

    def test_offsets_within_bounds(self, predictor):
        """start and end offsets must be within the text length."""
        text = "The patient was prescribed Aspirin for heart disease."
        entities = predictor.predict(text)
        for e in entities:
            assert 0 <= e.start < len(text)
            assert 0 < e.end <= len(text)
            assert e.start < e.end

    def test_offset_slices_match_text_field(self, predictor):
        """text[start:end] must equal entity.text."""
        text = "Aspirin can reduce the risk of heart disease."
        entities = predictor.predict(text)
        for e in entities:
            assert text[e.start:e.end] == e.text

    def test_empty_string_raises_or_returns_empty(self, predictor):
        """Empty or whitespace input should not crash."""
        result = predictor.predict(" ")
        assert isinstance(result, list)


# ===========================================================================
# 4. predict_batch
# ===========================================================================

class TestPredictBatch:
    """Tests for batch inference."""

    def test_returns_list_of_lists(self, predictor):
        texts = ["Aspirin for headache.", "Metformin for diabetes."]
        results = predictor.predict_batch(texts)
        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, list)

    def test_single_item_batch(self, predictor):
        text = "Aspirin treats headache."
        batch_result = predictor.predict_batch([text])
        single_result = predictor.predict(text)
        assert len(batch_result) == 1
        # Both should return same number of entities
        assert len(batch_result[0]) == len(single_result)

    def test_batch_vs_single_consistency(self, predictor):
        """Entities from batch mode should match single-call results."""
        texts = [
            "Aspirin is a chemical.",
            "Diabetes is a disease.",
            "No entities expected maybe.",
        ]
        batch_results = predictor.predict_batch(texts)
        for text, batch_ents in zip(texts, batch_results):
            single_ents = predictor.predict(text)
            assert len(batch_ents) == len(single_ents)
            for b, s in zip(batch_ents, single_ents):
                assert b.text == s.text
                assert b.label == s.label
                assert b.start == s.start
                assert b.end == s.end

    def test_empty_batch(self, predictor):
        results = predictor.predict_batch([])
        assert results == []
