from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import flwr as fl
import numpy as np


@dataclass(slots=True)
class RagDocument:
    doc_id: str
    embedding: Sequence[float]
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RagState:
    doc_ids: list[str]
    embeddings: np.ndarray
    texts: list[str]
    metadata: list[Mapping[str, Any]]


class RagIndex:
    """In-memory RAG index used to synchronize documents between clients."""

    def __init__(self, embedding_dim: int) -> None:
        self._embedding_dim = embedding_dim
        self._doc_ids: list[str] = []
        self._texts: list[str] = []
        self._metadata: list[Mapping[str, Any]] = []
        self._embeddings = np.empty((0, embedding_dim), dtype=np.float32)

    @property
    def size(self) -> int:
        return len(self._doc_ids)

    def add_documents(self, documents: Sequence[RagDocument]) -> None:
        if not documents:
            return
        for doc in documents:
            if doc.doc_id in self._doc_ids:
                continue
            self._doc_ids.append(doc.doc_id)
            self._texts.append(doc.text)
            self._metadata.append(doc.metadata)
            embedding = np.asarray(doc.embedding, dtype=np.float32).reshape(
                1, -1,
            )
            if embedding.shape[1] != self._embedding_dim:
                raise ValueError("Embedding dimension mismatch for RAG index")
            self._embeddings = np.vstack([self._embeddings, embedding])

    def merge_state(self, state: RagState) -> None:
        if state.doc_ids:
            for doc_id, text, metadata, embedding in zip(
                state.doc_ids,
                state.texts,
                state.metadata,
                state.embeddings,
                strict=False,
            ):
                self.add_documents(
                    [
                        RagDocument(
                            doc_id=doc_id,
                            text=text,
                            embedding=embedding.tolist(),
                            metadata=metadata,
                        ),
                    ],
                )

    def to_state(self) -> RagState:
        return RagState(
            doc_ids=list(self._doc_ids),
            embeddings=self._embeddings.copy(),
            texts=list(self._texts),
            metadata=list(self._metadata),
        )


def _serialize_state(state: RagState) -> bytes:
    payload = {
        "doc_ids": state.doc_ids,
        "texts": state.texts,
        "metadata": state.metadata,
        "embeddings": state.embeddings.tolist(),
    }
    return json.dumps(payload).encode("utf-8")


def _deserialize_state(payload: bytes) -> RagState:
    data = json.loads(payload.decode("utf-8"))
    embeddings = np.asarray(data.get("embeddings", []), dtype=np.float32)
    if embeddings.ndim == 1 and embeddings.size:
        embeddings = embeddings.reshape(1, -1)
    return RagState(
        doc_ids=list(data.get("doc_ids", [])),
        embeddings=embeddings,
        texts=list(data.get("texts", [])),
        metadata=list(data.get("metadata", [])),
    )


def state_to_parameters(state: RagState) -> fl.common.Parameters:
    payload = _serialize_state(state)
    return fl.common.Parameters(tensors=[payload], tensor_type="bytes")


def parameters_to_state(parameters: fl.common.Parameters) -> RagState | None:
    if not parameters.tensors:
        return None
    payload = parameters.tensors[0]
    return _deserialize_state(payload)
