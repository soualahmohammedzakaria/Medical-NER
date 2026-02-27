"""
ner_model.py
------------
Main NER model definitions.

Responsibilities:
 - Wrap a pre-trained transformer (BERT, BioBERT, PubMedBERT, etc.)
   with a token-classification head.
 - Optional CRF layer on top for structured decoding.
 - Expose a unified forward() / predict() interface.
"""
