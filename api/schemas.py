"""
api/schemas.py

Pydantic models for the Medical-NER prediction API.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Input payload: a raw text string to run NER on."""

    text: str = Field(
        ...,
        min_length=1,
        description="Raw clinical / biomedical text to analyse.",
        json_schema_extra={"example": "Aspirin can reduce the risk of heart disease."},
    )


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------

class EntitySpan(BaseModel):
    """A single detected entity."""

    text: str = Field(
        ...,
        description="Surface text of the entity as it appears in the input.",
    )
    label: str = Field(
        ...,
        description="Entity type, e.g. Chemical or Disease.",
    )
    start: int = Field(
        ...,
        ge=0,
        description="Character-level start offset (inclusive) in the input text.",
    )
    end: int = Field(
        ...,
        ge=0,
        description="Character-level end offset (exclusive) in the input text.",
    )


class PredictResponse(BaseModel):
    """Output payload: the original text plus all detected entities."""

    text: str = Field(
        ...,
        description="The input text that was analysed.",
    )
    entities: List[EntitySpan] = Field(
        default_factory=list,
        description="List of detected entity spans.",
    )
