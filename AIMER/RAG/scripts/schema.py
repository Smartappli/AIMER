from enum import Enum

from pydantic import BaseModel, Field


class DocType(str, Enum):
    """Supported document types.

    Values are stored/serialized as strings (subclassing `str` helps with JSON
    serialization and compatibility with vector DB metadata filters).
    """

    WE = "Wallonie_Elevages"


class ChunkMetadata(BaseModel):
    """Metadata attached to a document chunk.

    All fields are optional so the model can represent partial metadata and can
    be used as structured output from an LLM (fields not mentioned stay `None`).

    Attributes:
        doc_type: Document type/category.
        doc_month: Document month (1-12).
        doc_year: Document year (e.g. 2024).

    """

    doc_type: DocType | None = Field(default=None)
    doc_month: int | None = Field(default=None)
    doc_year: int | None = Field(default=None)


model_config = {"use_enum_values": True}
