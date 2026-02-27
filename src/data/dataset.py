"""
dataset.py
----------
Custom PyTorch / HuggingFace Dataset classes for NER corpora.

Responsibilities:
 - Load BIO/IOB-tagged data from CoNLL, JSON, or CSV formats.
 - Tokenise text and align entity labels with sub-word tokens.
 - Return input_ids, attention_mask, and label tensors.
"""
