"""
trainer.py

Training orchestration for Medical-NER using the HuggingFace Trainer.

Features:
  AdamW optimizer with linear warmup scheduler (via TrainingArguments).
  Early stopping on validation F1 with configurable patience.
  Best model checkpoint saving.
  Weights & Biases logging (falls back to CSV/TensorBoard if W&B is off).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.data.dataset import load_bc5cdr
from src.models.ner_model import build_model, build_tokenizer
from src.training.metrics import compute_metrics
from src.utils.helpers import load_config, seed_everything


# ---------------------------------------------------------------------------
# TrainConfig: populated from config.yaml, overridable via CLI
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """All tuneable knobs in one place."""

    model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    max_length: int = 512

    # optimiser
    learning_rate: float = 3e-5
    weight_decay: float = 0.01

    # scheduler
    warmup_steps: int = 500

    # training
    epochs: int = 5
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    fp16: bool = True

    # early stopping
    early_stopping_patience: int = 3

    # checkpointing
    output_dir: str = "outputs/models"
    logging_dir: str = "outputs/logs"
    save_total_limit: int = 2

    # logging
    use_wandb: bool = False
    wandb_project: str = "medical-ner"
    log_every_n_steps: int = 50
    csv_log: bool = True

    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str = "config/config.yaml") -> "TrainConfig":
        """Build a TrainConfig from a YAML file."""
        raw = load_config(path)
        t = raw.get("training", {})
        m = raw.get("model", {})
        d = raw.get("data", {})
        p = raw.get("project", {})
        lg = raw.get("logging", {})
        return cls(
            model_name=m.get("name", cls.model_name),
            max_length=d.get("max_seq_length", cls.max_length),
            learning_rate=t.get("learning_rate", cls.learning_rate),
            weight_decay=t.get("weight_decay", cls.weight_decay),
            warmup_steps=t.get("warmup_steps", cls.warmup_steps),
            epochs=t.get("epochs", cls.epochs),
            batch_size=t.get("batch_size", cls.batch_size),
            gradient_accumulation_steps=t.get("gradient_accumulation_steps", cls.gradient_accumulation_steps),
            fp16=t.get("fp16", cls.fp16),
            early_stopping_patience=t.get("early_stopping_patience", cls.early_stopping_patience),
            output_dir=t.get("output_dir", cls.output_dir),
            logging_dir=t.get("logging_dir", cls.logging_dir),
            save_total_limit=t.get("save_total_limit", cls.save_total_limit),
            log_every_n_steps=t.get("log_every_n_steps", cls.log_every_n_steps),
            use_wandb=lg.get("use_wandb", cls.use_wandb),
            wandb_project=lg.get("wandb_project", cls.wandb_project),
            csv_log=lg.get("csv_log", cls.csv_log),
            seed=p.get("seed", cls.seed),
        )


# ---------------------------------------------------------------------------
# Trainer builder
# ---------------------------------------------------------------------------

def build_training_args(cfg: TrainConfig) -> TrainingArguments:
    """Translate our simple TrainConfig into HuggingFace TrainingArguments."""

    # Decide on report_to depending on user preference.
    report_to = []
    if cfg.csv_log:
        report_to.append("tensorboard")
    if cfg.use_wandb:
        report_to.append("wandb")
    if not report_to:
        report_to = ["none"]

    # Only enable fp16 when a CUDA GPU is actually available.
    use_fp16 = cfg.fp16 and torch.cuda.is_available()

    # Point TensorBoard at the logging directory via env var
    # (logging_dir kwarg is deprecated in transformers >= 4.47).
    if cfg.logging_dir:
        os.environ["TENSORBOARD_LOGGING_DIR"] = cfg.logging_dir

    return TrainingArguments(
        output_dir=cfg.output_dir,

        # training
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size * 2,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        fp16=use_fp16,

        # optimiser / scheduler
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        optim="adamw_torch",

        # evaluation / checkpointing
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        # logging
        logging_steps=cfg.log_every_n_steps,
        report_to=report_to,
        run_name=cfg.wandb_project,

        # data loading
        dataloader_pin_memory=torch.cuda.is_available(),

        # reproducibility
        seed=cfg.seed,
        data_seed=cfg.seed,
    )


def train(cfg: TrainConfig | None = None) -> Trainer:
    """End-to-end training run. Returns the Trainer for further inspection.

    Steps
    -----
    1. Load tokenizer and tokenized BC5CDR splits.
    2. Build BiomedBERT + classification head.
    3. Construct HuggingFace Trainer with early stopping callback.
    4. Call trainer.train().
    5. Save the best model and tokenizer to output_dir/best.
    """
    if cfg is None:
        cfg = TrainConfig()

    # seed all RNGs for reproducibility
    seed_everything(cfg.seed)

    # suppress TensorFlow oneDNN info messages (irrelevant when using PyTorch)
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    # optional W&B setup
    if cfg.use_wandb:
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)
    else:
        os.environ["WANDB_DISABLED"] = "true"

    # data
    tokenizer = build_tokenizer(cfg.model_name)
    datasets = load_bc5cdr(tokenizer, max_length=cfg.max_length)

    # model
    model = build_model(model_name=cfg.model_name)

    # training args
    training_args = build_training_args(cfg)

    # callbacks
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=cfg.early_stopping_patience,
        ),
    ]

    # trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # train
    trainer.train()

    # save best model + tokenizer
    best_dir = Path(cfg.output_dir) / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print(f"\nBest model saved to {best_dir}")

    return trainer