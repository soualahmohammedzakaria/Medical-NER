"""
api/schemas.py

Pydantic models for the Medical-NER prediction API.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

MAX_TEXT_LENGTH = 10_000   # chars -- reject overly long inputs early


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Input payload: a raw text string to run NER on."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        description="Raw clinical / biomedical text to analyse.",
        json_schema_extra={
            "example": "Aspirin can reduce the risk of heart disease.",
        },
    )

    @model_validator(mode="after")
    def strip_text(self) -> "PredictRequest":
        """Remove leading / trailing whitespace."""
        self.text = self.text.strip()
        if not self.text:
            raise ValueError("text must not be blank after stripping whitespace.")
        return self


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


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    """Standard error envelope returned by all error handlers."""

    error: str = Field(..., description="Short error code.")
    message: str = Field(..., description="Human-readable detail.")
