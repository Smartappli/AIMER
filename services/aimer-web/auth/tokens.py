# Copyright (c) 2026 AIMER contributors.
"""Helpers for one-time authentication tokens."""

from __future__ import annotations

import hashlib
import secrets


def hash_token(token: str) -> str:
    """
    Return the stable hash stored for an opaque one-time token.

    The raw token is high-entropy and is only sent to the user. Persisting a
    hash limits impact if the database is disclosed.

    Args:
        token: Raw token value.

    Returns:
        SHA-256 hexadecimal digest.

    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_token_pair() -> tuple[str, str]:
    """
    Generate a raw token and its persisted hash.

    Returns:
        Tuple of ``(raw_token, hashed_token)``.

    """
    raw_token = secrets.token_urlsafe(32)
    return raw_token, hash_token(raw_token)
