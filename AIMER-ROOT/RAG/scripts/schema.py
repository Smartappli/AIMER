# Copyright (c) 2026 AIMER contributors.
"""Schema definitions for RAG chunk metadata."""

from enum import StrEnum

from pydantic import BaseModel, Field


class DocType(StrEnum):
    """
    Supported document types.

    Values are stored/serialized as strings (subclassing `str` helps with JSON
    serialization and compatibility with vector DB metadata filters).
    """

    WE = "Wallonie_Elevages"


class ChunkMetadata(BaseModel):
    """
    Metadata attached to a document chunk.

    All fields are optional so the model can represent partial metadata and can
    be used as structured output from an LLM (fields not mentioned stay `None`).

    Attributes:
        doc_type: Document type/category.
        doc_month: Document month (1-12).
        doc_year: Document year (e.g. 2024).
        omop_condition_concept_ids: OMOP condition concept identifiers.
        omop_procedure_concept_ids: OMOP procedure concept identifiers.
        omop_measurement_concept_ids: OMOP measurement concept identifiers.
        omop_modality_concept_ids: OMOP modality concept identifiers.
        snomed_ct_codes: SNOMED CT identifiers linked to the extracted concepts.

    """

    doc_type: DocType | None = Field(default=None)
    doc_month: int | None = Field(default=None)
    doc_year: int | None = Field(default=None)
    omop_condition_concept_ids: list[int] | None = Field(default=None)
    omop_procedure_concept_ids: list[int] | None = Field(default=None)
    omop_measurement_concept_ids: list[int] | None = Field(default=None)
    omop_modality_concept_ids: list[int] | None = Field(default=None)
    snomed_ct_codes: list[str] | None = Field(default=None)


model_config = {"use_enum_values": True}
