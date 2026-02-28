"""
scripts/predict.py

CLI entry-point for Medical-NER inference.

Usage examples
--------------
  # Predict on a single sentence
  python -m scripts.predict --checkpoint outputs/models/best \
      --input "Aspirin can reduce the risk of heart disease."

  # Predict on every line of a text file
  python -m scripts.predict --checkpoint outputs/models/best \
      --file data/raw/samples.txt

  # Save results as JSON
  python -m scripts.predict --checkpoint outputs/models/best \
      --input "Metformin treats diabetes." --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.predict import NERPredictor


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _print_entities(text: str, entities: list) -> None:
    """Print a single text and its extracted entities to stdout."""
    print(f"\nText: {text}")
    if not entities:
        print("  (no entities found)")
        return
    for ent in entities:
        print(f"  [{ent.label}] \"{ent.text}\"  (chars {ent.start}-{ent.end})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Medical-NER inference on raw text.",
    )
    p.add_argument(
        "--checkpoint", type=str, default="outputs/models/best",
        help="Path to the saved model directory (default: outputs/models/best).",
    )
    p.add_argument(
        "--input", "-i", type=str, default=None,
        help="A single text string to run prediction on.",
    )
    p.add_argument(
        "--file", "-f", type=str, default=None,
        help="Path to a text file. Each non-empty line is treated as one sample.",
    )
    p.add_argument(
        "--device", type=str, default="auto",
        help="Device for inference: cpu, cuda, mps, or auto (default: auto).",
    )
    p.add_argument(
        "--output", "-o", type=str, default=None,
        help="Optional path to save results as a JSON file.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.input is None and args.file is None:
        print("ERROR: provide --input or --file (or both).", file=sys.stderr)
        sys.exit(1)

    # Build predictor
    predictor = NERPredictor(
        checkpoint_dir=args.checkpoint,
        device=args.device,
    )
    print(f"Model loaded from: {args.checkpoint}")

    # Collect texts
    texts: list[str] = []
    if args.input is not None:
        texts.append(args.input)
    if args.file is not None:
        filepath = Path(args.file)
        if not filepath.is_file():
            print(f"ERROR: file not found: {filepath}", file=sys.stderr)
            sys.exit(1)
        lines = filepath.read_text(encoding="utf-8").splitlines()
        texts.extend(line.strip() for line in lines if line.strip())

    # Run inference
    all_results: list[dict] = []
    for text in texts:
        entities = predictor.predict(text)
        _print_entities(text, entities)
        all_results.append({
            "text": text,
            "entities": [e.to_dict() for e in entities],
        })

    # Optionally save to JSON
    if args.output is not None:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(all_results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()