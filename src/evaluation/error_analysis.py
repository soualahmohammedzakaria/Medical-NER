"""
error_analysis.py

Detailed error analysis for Medical-NER predictions.

Analyses:
  1. Top-N most common false positives  (predicted entity not in gold).
  2. Top-N most common false negatives  (gold entity missed by model).
  3. Boundary errors (partial span overlap between pred and gold).
  4. Negation errors (negated entities incorrectly tagged as positive).

All helpers work on decoded IOB2 tag sequences (list of sentences, each a
list of tag strings) plus the corresponding word-level tokens.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Span extraction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Span:
    """A single entity span with its surface text, type, and position."""
    entity_type: str       # e.g. "Chemical", "Disease"
    start: int             # token-level start index (inclusive)
    end: int               # token-level end index (exclusive)
    text: str              # space-joined surface tokens

    def overlaps(self, other: "Span") -> bool:
        """True if the two spans share at least one token position."""
        return self.start < other.end and other.start < self.end


def extract_spans(tags: List[str], tokens: List[str]) -> List[Span]:
    """Extract entity spans from a single IOB2 tag sequence.

    Parameters
    ----------
    tags : list of str
        IOB2 tags for one sentence (e.g. ["O", "B-Chemical", "I-Chemical", ...]).
    tokens : list of str
        The corresponding word-level tokens.

    Returns
    -------
    list of Span
    """
    spans: List[Span] = []
    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag.startswith("B-"):
            etype = tag[2:]
            start = i
            i += 1
            # consume following I- tags of the same type
            while i < len(tags) and tags[i] == f"I-{etype}":
                i += 1
            text = " ".join(tokens[start:i])
            spans.append(Span(entity_type=etype, start=start, end=i, text=text))
        else:
            i += 1
    return spans


# ---------------------------------------------------------------------------
# 1. False positives
# ---------------------------------------------------------------------------

def find_false_positives(
    pred_tags: List[List[str]],
    gold_tags: List[List[str]],
    all_tokens: List[List[str]],
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """Return the top-N most common false positive entity texts.

    A false positive is a predicted span that does not exactly match any
    gold span (same type, same start, same end).
    """
    fp_counter: Counter = Counter()

    for preds, golds, tokens in zip(pred_tags, gold_tags, all_tokens):
        pred_spans = extract_spans(preds, tokens)
        gold_spans_set: Set[Tuple[str, int, int]] = {
            (s.entity_type, s.start, s.end) for s in extract_spans(golds, tokens)
        }
        for sp in pred_spans:
            if (sp.entity_type, sp.start, sp.end) not in gold_spans_set:
                fp_counter[(sp.text, sp.entity_type)] += 1

    return [
        {"text": text, "type": etype, "count": count}
        for (text, etype), count in fp_counter.most_common(top_n)
    ]


# ---------------------------------------------------------------------------
# 2. False negatives
# ---------------------------------------------------------------------------

def find_false_negatives(
    pred_tags: List[List[str]],
    gold_tags: List[List[str]],
    all_tokens: List[List[str]],
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """Return the top-N most common false negative entity texts.

    A false negative is a gold span that has no exact match in the
    predicted spans.
    """
    fn_counter: Counter = Counter()

    for preds, golds, tokens in zip(pred_tags, gold_tags, all_tokens):
        pred_spans_set: Set[Tuple[str, int, int]] = {
            (s.entity_type, s.start, s.end) for s in extract_spans(preds, tokens)
        }
        for sp in extract_spans(golds, tokens):
            if (sp.entity_type, sp.start, sp.end) not in pred_spans_set:
                fn_counter[(sp.text, sp.entity_type)] += 1

    return [
        {"text": text, "type": etype, "count": count}
        for (text, etype), count in fn_counter.most_common(top_n)
    ]


# ---------------------------------------------------------------------------
# 3. Boundary errors (partial overlap)
# ---------------------------------------------------------------------------

def find_boundary_errors(
    pred_tags: List[List[str]],
    gold_tags: List[List[str]],
    all_tokens: List[List[str]],
    max_examples: int = 10,
) -> List[Dict[str, Any]]:
    """Find examples where a predicted span partially overlaps a gold span.

    A boundary error is defined as: same entity type, overlapping token
    positions, but different start or end (i.e. not an exact match).
    """
    examples: List[Dict[str, Any]] = []

    for sent_idx, (preds, golds, tokens) in enumerate(
        zip(pred_tags, gold_tags, all_tokens)
    ):
        pred_spans = extract_spans(preds, tokens)
        gold_spans = extract_spans(golds, tokens)

        for ps in pred_spans:
            for gs in gold_spans:
                if ps.entity_type != gs.entity_type:
                    continue
                # overlapping but not identical = boundary error
                if ps.overlaps(gs) and (ps.start != gs.start or ps.end != gs.end):
                    examples.append({
                        "sentence_idx": sent_idx,
                        "gold_text": gs.text,
                        "gold_span": [gs.start, gs.end],
                        "pred_text": ps.text,
                        "pred_span": [ps.start, ps.end],
                        "type": ps.entity_type,
                        "context": " ".join(tokens),
                    })
                    if len(examples) >= max_examples:
                        return examples

    return examples


# ---------------------------------------------------------------------------
# 4. Negation errors
# ---------------------------------------------------------------------------

# Simple lexical patterns that suggest a negated clinical context.
_NEGATION_CUES = re.compile(
    r"\b(no|not|without|absence|absent|deny|denies|denied|negative|"
    r"neither|nor|never|none|rule out|rules out|ruled out|unlikely|"
    r"free of|lack of|lacks|exclude|excludes|excluded)\b",
    re.IGNORECASE,
)

# Window of tokens before an entity span to scan for negation cues.
_NEG_WINDOW = 5


def find_negation_errors(
    pred_tags: List[List[str]],
    gold_tags: List[List[str]],
    all_tokens: List[List[str]],
    max_examples: int = 10,
) -> List[Dict[str, Any]]:
    """Find predicted entities that sit in a negated context.

    A negation error is a predicted entity span whose preceding tokens
    (within a small window) contain a negation cue AND the gold label
    for those tokens is O (meaning the entity should not have been tagged).

    This is a heuristic based on lexical cue words; it will not catch
    all negation patterns but covers the most common clinical ones.
    """
    examples: List[Dict[str, Any]] = []

    for sent_idx, (preds, golds, tokens) in enumerate(
        zip(pred_tags, gold_tags, all_tokens)
    ):
        pred_spans = extract_spans(preds, tokens)
        gold_spans_set: Set[Tuple[str, int, int]] = {
            (s.entity_type, s.start, s.end) for s in extract_spans(golds, tokens)
        }

        for sp in pred_spans:
            # only consider false positives (not in gold)
            if (sp.entity_type, sp.start, sp.end) in gold_spans_set:
                continue

            # look at the window before the span for negation cues
            window_start = max(0, sp.start - _NEG_WINDOW)
            window_text = " ".join(tokens[window_start : sp.start])

            if _NEGATION_CUES.search(window_text):
                examples.append({
                    "sentence_idx": sent_idx,
                    "entity_text": sp.text,
                    "type": sp.entity_type,
                    "negation_window": window_text,
                    "context": " ".join(tokens),
                })
                if len(examples) >= max_examples:
                    return examples

    return examples


# ---------------------------------------------------------------------------
# Full error analysis pipeline
# ---------------------------------------------------------------------------

def run_error_analysis(
    pred_tags: List[List[str]],
    gold_tags: List[List[str]],
    all_tokens: List[List[str]],
    top_n: int = 10,
    output_path: str | None = None,
) -> Dict[str, Any]:
    """Run all four error analyses and return/save a combined report.

    Parameters
    ----------
    pred_tags, gold_tags : list of list of str
        Decoded IOB2 tag sequences (one inner list per sentence).
    all_tokens : list of list of str
        Original word-level tokens aligned with the tag sequences.
    top_n : int
        How many items to show for FP / FN lists.
    output_path : str or None
        If given, write the report as JSON to this path.

    Returns
    -------
    dict  with keys: false_positives, false_negatives, boundary_errors,
          negation_errors.
    """
    print("\n" + "=" * 55)
    print("  Error Analysis")
    print("=" * 55)

    # 1. False positives
    fps = find_false_positives(pred_tags, gold_tags, all_tokens, top_n)
    print(f"\n[1] Top {top_n} false positives:")
    for i, fp in enumerate(fps, 1):
        print(f"  {i:2d}. \"{fp['text']}\" ({fp['type']}) x{fp['count']}")

    # 2. False negatives
    fns = find_false_negatives(pred_tags, gold_tags, all_tokens, top_n)
    print(f"\n[2] Top {top_n} false negatives:")
    for i, fn in enumerate(fns, 1):
        print(f"  {i:2d}. \"{fn['text']}\" ({fn['type']}) x{fn['count']}")

    # 3. Boundary errors
    boundary = find_boundary_errors(pred_tags, gold_tags, all_tokens, top_n)
    print(f"\n[3] Boundary errors ({len(boundary)} examples):")
    for i, be in enumerate(boundary, 1):
        print(f"  {i:2d}. gold=\"{be['gold_text']}\" pred=\"{be['pred_text']}\" "
              f"({be['type']})")
        print(f"      gold_span={be['gold_span']}  pred_span={be['pred_span']}")

    # 4. Negation errors
    negation = find_negation_errors(pred_tags, gold_tags, all_tokens, top_n)
    print(f"\n[4] Negation errors ({len(negation)} examples):")
    for i, ne in enumerate(negation, 1):
        print(f"  {i:2d}. \"{ne['entity_text']}\" ({ne['type']})")
        print(f"      cue window: \"{ne['negation_window']}\"")

    # assemble report
    report: Dict[str, Any] = {
        "false_positives": fps,
        "false_negatives": fns,
        "boundary_errors": boundary,
        "negation_errors": negation,
    }

    # save to JSON
    if output_path is None:
        out_dir = Path("outputs/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / "error_analysis.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nError analysis saved to {output_path}")

    return report