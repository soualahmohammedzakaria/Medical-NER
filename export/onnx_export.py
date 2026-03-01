"""
export/onnx_export.py

Export the fine-tuned BiomedBERT token-classification model to ONNX format,
then benchmark inference latency (PyTorch vs ONNX Runtime) on 100 sample
sentences.  Prints a summary table and saves results to a JSON file.

Usage
-----
  python -m export.onnx_export                              # defaults
  python -m export.onnx_export --checkpoint outputs/models/best
  python -m export.onnx_export --checkpoint outputs/models/best \
      --output-dir outputs/onnx --num-samples 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.ner_model import id2label


# ---------------------------------------------------------------------------
# Sample sentences for benchmarking
# ---------------------------------------------------------------------------

_SAMPLE_SENTENCES = [
    "The patient was prescribed Metformin for type 2 diabetes.",
    "Aspirin is commonly used as an anti-inflammatory drug.",
    "Ibuprofen may cause gastrointestinal bleeding in some patients.",
    "Doxorubicin is a chemotherapy agent used to treat breast cancer.",
    "Acetaminophen overdose can lead to acute liver failure.",
    "Warfarin therapy requires regular monitoring of INR levels.",
    "The study evaluated the efficacy of Remdesivir against COVID-19.",
    "Cisplatin-induced nephrotoxicity is a major clinical concern.",
    "Patients with hypertension were treated with Lisinopril.",
    "Cyclosporine is used to prevent organ transplant rejection.",
    "Penicillin allergy is one of the most commonly reported drug allergies.",
    "Omeprazole is a proton pump inhibitor used for gastric ulcers.",
    "Rituximab has shown efficacy in treating non-Hodgkin lymphoma.",
    "Insulin resistance is a hallmark of metabolic syndrome.",
    "Tamoxifen is prescribed for estrogen-receptor-positive breast cancer.",
    "Corticosteroids such as Prednisone may cause osteoporosis.",
    "Carbamazepine is used to manage epilepsy and trigeminal neuralgia.",
    "The hepatotoxicity of Isoniazid requires liver function monitoring.",
    "Atorvastatin reduces low-density lipoprotein cholesterol levels.",
    "Morphine is an opioid analgesic for moderate to severe pain.",
]


def _build_sample_texts(n: int) -> list[str]:
    """Return *n* sample sentences by cycling through the seed list."""
    return [_SAMPLE_SENTENCES[i % len(_SAMPLE_SENTENCES)] for i in range(n)]


# ---------------------------------------------------------------------------
# 1. ONNX export
# ---------------------------------------------------------------------------

def export_to_onnx(
    checkpoint_dir: str,
    output_path: str,
    opset_version: int = 14,
) -> Path:
    """Export a HuggingFace token-classification model to ONNX.

    Parameters
    ----------
    checkpoint_dir : str
        Path to the saved model directory (contains config.json, model.safetensors).
    output_path : str
        Destination file path for the .onnx model.
    opset_version : int
        ONNX opset version (default: 14).

    Returns
    -------
    Path
        The resolved path to the exported .onnx file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from: {checkpoint_dir}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(checkpoint_dir)
    model.eval()

    # Create dummy input
    dummy_text = "Aspirin treats headache."
    dummy = tokenizer(dummy_text, return_tensors="pt")
    input_ids = dummy["input_ids"]
    attention_mask = dummy["attention_mask"]

    print(f"Exporting to ONNX (opset {opset_version}) -> {out}")
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        str(out),
        opset_version=opset_version,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch", 1: "seq_len"},
        },
    )

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"Export complete. File size: {size_mb:.1f} MB")
    return out


# ---------------------------------------------------------------------------
# 2. Benchmark helpers
# ---------------------------------------------------------------------------

def _benchmark_pytorch(
    model: torch.nn.Module,
    encodings: list[dict],
    device: torch.device,
) -> list[float]:
    """Run PyTorch inference and return per-sample latencies in ms."""
    model.to(device)
    model.eval()
    latencies: list[float] = []

    # Warm-up (3 iterations)
    for enc in encodings[:3]:
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask)

    for enc in encodings:
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask)

        if device.type == "cuda":
            torch.cuda.synchronize()

        latencies.append((time.perf_counter() - t0) * 1000)

    return latencies


def _benchmark_onnx(
    onnx_path: str,
    encodings: list[dict],
) -> list[float]:
    """Run ONNX Runtime inference and return per-sample latencies in ms."""
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")

    session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
    active_provider = session.get_providers()[0]
    print(f"  ONNX Runtime provider: {active_provider}")

    latencies: list[float] = []

    # Warm-up
    for enc in encodings[:3]:
        feed = {
            "input_ids": enc["input_ids"].numpy(),
            "attention_mask": enc["attention_mask"].numpy(),
        }
        session.run(None, feed)

    for enc in encodings:
        feed = {
            "input_ids": enc["input_ids"].numpy(),
            "attention_mask": enc["attention_mask"].numpy(),
        }
        t0 = time.perf_counter()
        session.run(None, feed)
        latencies.append((time.perf_counter() - t0) * 1000)

    return latencies


# ---------------------------------------------------------------------------
# 3. Report
# ---------------------------------------------------------------------------

def _summarise(latencies: list[float]) -> dict:
    """Compute summary statistics for a list of latencies (ms)."""
    arr = np.array(latencies)
    return {
        "mean_ms": round(float(arr.mean()), 2),
        "median_ms": round(float(np.median(arr)), 2),
        "std_ms": round(float(arr.std()), 2),
        "min_ms": round(float(arr.min()), 2),
        "max_ms": round(float(arr.max()), 2),
        "p90_ms": round(float(np.percentile(arr, 90)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "samples": len(latencies),
    }


def _print_table(pytorch_stats: dict, onnx_stats: dict) -> None:
    """Pretty-print a comparison table to stdout."""
    header = f"{'Metric':<16} {'PyTorch':>12} {'ONNX RT':>12} {'Speedup':>10}"
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for key in ("mean_ms", "median_ms", "std_ms", "min_ms", "max_ms",
                "p90_ms", "p99_ms"):
        pt = pytorch_stats[key]
        ox = onnx_stats[key]
        speedup = pt / ox if ox > 0 else float("inf")
        label = key.replace("_ms", " (ms)")
        print(f"{label:<16} {pt:>12.2f} {ox:>12.2f} {speedup:>9.2f}x")
    print(sep)


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def run(
    checkpoint_dir: str = "outputs/models/best",
    output_dir: str = "outputs/onnx",
    num_samples: int = 100,
    opset_version: int = 14,
) -> dict:
    """Full pipeline: export, benchmark, report.

    Returns
    -------
    dict
        Benchmark results with pytorch and onnx summary stats.
    """
    out_dir = Path(output_dir)
    onnx_path = out_dir / "model.onnx"

    # -- Export ------------------------------------------------------------
    export_to_onnx(checkpoint_dir, str(onnx_path), opset_version)

    # -- Prepare encodings -------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    texts = _build_sample_texts(num_samples)
    encodings = [
        tokenizer(t, return_tensors="pt", truncation=True, max_length=512)
        for t in texts
    ]

    # -- Load PyTorch model ------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForTokenClassification.from_pretrained(checkpoint_dir)
    print(f"\nBenchmarking on device: {device}  |  samples: {num_samples}")

    # -- Benchmark PyTorch -------------------------------------------------
    print("\n  Running PyTorch benchmark ...")
    pt_latencies = _benchmark_pytorch(model, encodings, device)
    pt_stats = _summarise(pt_latencies)

    # -- Benchmark ONNX Runtime --------------------------------------------
    print("  Running ONNX Runtime benchmark ...")
    ox_latencies = _benchmark_onnx(str(onnx_path), encodings)
    ox_stats = _summarise(ox_latencies)

    # -- Report ------------------------------------------------------------
    _print_table(pt_stats, ox_stats)

    results = {
        "checkpoint": checkpoint_dir,
        "onnx_path": str(onnx_path),
        "device": str(device),
        "num_samples": num_samples,
        "pytorch": pt_stats,
        "onnx_runtime": ox_stats,
        "speedup_mean": round(pt_stats["mean_ms"] / ox_stats["mean_ms"], 2)
        if ox_stats["mean_ms"] > 0 else None,
    }

    # -- Save --------------------------------------------------------------
    results_path = out_dir / "benchmark_results.json"
    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nResults saved to: {results_path}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export BiomedBERT NER to ONNX and benchmark latency.",
    )
    p.add_argument("--checkpoint", type=str, default="outputs/models/best",
                    help="Path to the fine-tuned model directory.")
    p.add_argument("--output-dir", type=str, default="outputs/onnx",
                    help="Directory for ONNX model and results JSON.")
    p.add_argument("--num-samples", type=int, default=100,
                    help="Number of sample sentences for benchmarking.")
    p.add_argument("--opset", type=int, default=14,
                    help="ONNX opset version (default: 14).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(
        checkpoint_dir=args.checkpoint,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        opset_version=args.opset,
    )


if __name__ == "__main__":
    main()