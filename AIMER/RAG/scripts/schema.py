from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class DocType(str, Enum):
    WE = "Wallonie_Elevages"


class ChunkMetadata(BaseModel):
    doc_type: Optional[DocType] = Field(default=None)

    doc_month: Optional[int] = Field(default=None)

    doc_year: Optional[int] = Field(default=None)


model_config = {"use_enum_values": True}
