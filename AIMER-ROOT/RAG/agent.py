# Copyright (c) 2026 AIMER contributors.
"""Shared Ollama chat model configuration for the RAG agent package."""

from langchain_ollama import ChatOllama

LLM_MODEL = ""

model = ChatOllama(model=LLM_MODEL, base_url="http://localhost:11434/")
