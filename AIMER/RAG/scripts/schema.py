from enum import Enum

from pydantic import BaseModel, Field


class DocType(str, Enum):
    WE = "Wallonie_Elevages"


class ChunkMetadata(BaseModel):
    doc_type: DocType | None = Field(default=None)

    doc_month: int | None = Field(default=None)

    doc_year: int | None = Field(default=None)


model_config = {"use_enum_values": True}
