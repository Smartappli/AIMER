# Copyright (c) 2026 AIMER contributors.
"""Shared Ollama chat model configuration for the RAG agent package."""

import os

from langchain_ollama import ChatOllama

LLM_MODEL = os.getenv("RAG_LLM_MODEL", "qwen3:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

model = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
