"""
predict.py

Inference pipeline for Medical-NER.

Loads a fine-tuned checkpoint, accepts raw text, tokenizes it,
runs a forward pass, decodes IOB2 tags back into entity spans,
and returns structured results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)

from src.models.ner_model import id2label
from src.utils.helpers import get_device


# ---------------------------------------------------------------------------
# Entity dataclass
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    """A single extracted entity."""
    text: str
    label: str       # e.g. "Chemical", "Disease"
    start: int       # character-level start offset in original text
    end: int         # character-level end offset in original text

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "label": self.label,
                "start": self.start, "end": self.end}


# ---------------------------------------------------------------------------
# NER Predictor
# ---------------------------------------------------------------------------

class NERPredictor:
    """Wraps a fine-tuned token-classification model for inference."""

    def __init__(
        self,
        checkpoint_dir: str,
        device: str = "auto",
    ) -> None:
        self.device = get_device(device)
        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            checkpoint_dir, use_fast=True,
        )
        self.model = AutoModelForTokenClassification.from_pretrained(checkpoint_dir)
        self.model.to(self.device)
        self.model.eval()

    # ---- public API -------------------------------------------------------

    def predict(self, text: str) -> List[Entity]:
        """Extract entities from a single string.

        Parameters
        ----------
        text : str
            Raw input text (e.g. a clinical sentence or paragraph).

        Returns
        -------
        list of Entity
        """
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,  # needed to map back to char positions
        )

        # offset_mapping is not a model input
        offset_mapping = encoding.pop("offset_mapping")[0]   # (seq_len, 2)
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

        pred_ids = torch.argmax(logits, dim=-1)[0].cpu().tolist()

        return self._decode_entities(text, pred_ids, offset_mapping)

    def predict_batch(self, texts: List[str]) -> List[List[Entity]]:
        """Extract entities from multiple strings at once."""
        return [self.predict(t) for t in texts]

    # ---- internal ---------------------------------------------------------

    @staticmethod
    def _decode_entities(
        text: str,
        pred_ids: List[int],
        offset_mapping: torch.Tensor,
    ) -> List[Entity]:
        """Convert predicted label ids + offsets into Entity objects.

        Merges consecutive B- / I- tags of the same type into a single span,
        then uses the offset mapping to recover character-level positions and
        the original surface text (preserving casing and whitespace).
        """
        entities: List[Entity] = []
        offsets = offset_mapping.tolist()  # list of (start_char, end_char)

        i = 0
        while i < len(pred_ids):
            tag = id2label.get(pred_ids[i], "O")

            if tag.startswith("B-"):
                etype = tag[2:]
                span_start_char = offsets[i][0]
                span_end_char = offsets[i][1]
                i += 1

                # absorb following I- tokens of the same type
                while i < len(pred_ids):
                    next_tag = id2label.get(pred_ids[i], "O")
                    if next_tag == f"I-{etype}":
                        span_end_char = offsets[i][1]
                        i += 1
                    else:
                        break

                # skip if the offsets point to special tokens (both 0)
                if span_start_char == 0 and span_end_char == 0:
                    continue

                entities.append(Entity(
                    text=text[span_start_char:span_end_char],
                    label=etype,
                    start=span_start_char,
                    end=span_end_char,
                ))
            else:
                i += 1

        return entities