"""
trainer.py
----------
Training orchestration for NER models.

Responsibilities:
 - Configure optimizer, scheduler, and mixed-precision settings.
 - Run the training loop with gradient accumulation.
 - Handle checkpointing, early stopping, and experiment logging
   (W&B / TensorBoard).
"""
