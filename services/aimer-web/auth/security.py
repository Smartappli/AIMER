# Copyright (c) 2026 AIMER contributors.
"""Authentication security controls and audit helpers."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from django.conf import settings
from django.contrib.auth.models import AnonymousUser
from django.core.cache import cache
from django.db import DatabaseError
from django.http import HttpRequest

from auth.models import SecurityAuditEvent

DEFAULT_LOGIN_FAILURE_LIMIT = 5
DEFAULT_LOGIN_WINDOW_SECONDS = 15 * 60
DEFAULT_LOGIN_LOCKOUT_SECONDS = 15 * 60
AUDIT_LOGGER = logging.getLogger("aimer.security.audit")
SENSITIVE_METADATA_FRAGMENTS = (
    "authorization",
    "cookie",
    "key",
    "password",
    "secret",
    "token",
)


def _setting_int(name: str, default: int) -> int:
    """Return an integer setting with a conservative fallback."""
    value = getattr(settings, name, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def client_ip(request: HttpRequest | None) -> str | None:
    """Return the direct peer IP address recorded by Django."""
    if request is None:
        return None
    value = request.META.get("REMOTE_ADDR")
    return str(value) if value else None


def _user_agent(request: HttpRequest | None) -> str:
    """Return a bounded user-agent string."""
    if request is None:
        return ""
    return str(request.META.get("HTTP_USER_AGENT", ""))[:1024]


def _path(request: HttpRequest | None) -> str:
    """Return a bounded request path."""
    if request is None:
        return ""
    return str(getattr(request, "path", ""))[:512]


def _actor(user: object | None) -> object | None:
    """Return a persisted user object or ``None`` for anonymous actors."""
    if user is None or isinstance(user, AnonymousUser):
        return None
    if bool(getattr(user, "is_authenticated", False)):
        return user
    return None


def _sanitize_metadata(value: Any) -> Any:
    """Return metadata with obvious secret-bearing fields redacted."""
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            lowered = key_str.lower()
            if any(fragment in lowered for fragment in SENSITIVE_METADATA_FRAGMENTS):
                redacted[key_str] = "[REDACTED]"
            else:
                redacted[key_str] = _sanitize_metadata(item)
        return redacted
    if isinstance(value, list):
        return [_sanitize_metadata(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_metadata(item) for item in value]
    if isinstance(value, str):
        return value[:1024]
    return value


def _audit_log_payload(
    event_type: str,
    *,
    request: HttpRequest | None,
    user: object | None,
    actor_identifier: str,
    metadata: dict[str, Any] | None,
    persisted: bool,
) -> dict[str, Any]:
    """Build a structured audit payload suitable for SIEM collection."""
    actor = _actor(user)
    return {
        "event_type": event_type,
        "actor_identifier": actor_identifier[:255],
        "user_id": getattr(actor, "pk", None),
        "ip_address": client_ip(request),
        "user_agent": _user_agent(request),
        "path": _path(request),
        "metadata": _sanitize_metadata(metadata or {}),
        "persisted": persisted,
    }


def _log_audit_event(payload: dict[str, Any]) -> None:
    """Emit a JSON audit event without interrupting request handling."""
    try:
        AUDIT_LOGGER.info(
            "security_audit_event",
            extra={"security_audit_event": payload},
        )
    except (TypeError, ValueError):
        AUDIT_LOGGER.info(
            "security_audit_serialization_failed",
            extra={
                "security_audit_event": {
                    "event_type": "security_audit.serialization_failed",
                    "persisted": payload.get("persisted", False),
                },
            },
        )


def audit_event(
    event_type: str,
    *,
    request: HttpRequest | None = None,
    user: object | None = None,
    actor_identifier: str = "",
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist a security audit event without raising into request handling."""
    safe_metadata = _sanitize_metadata(metadata or {})
    persisted = False
    try:
        SecurityAuditEvent.objects.create(
            event_type=event_type,
            user=_actor(user),
            actor_identifier=actor_identifier[:255],
            ip_address=client_ip(request),
            user_agent=_user_agent(request),
            path=_path(request),
            metadata=safe_metadata,
        )
        persisted = True
    except DatabaseError:
        # Audit must not make authentication unavailable. Infrastructure should
        # still alert on database write failures through normal error logging.
        persisted = False

    payload = _audit_log_payload(
        event_type,
        request=request,
        user=user,
        actor_identifier=actor_identifier,
        metadata=safe_metadata,
        persisted=persisted,
    )
    _log_audit_event(payload)


def _principal_hash(identifier: str, ip_address: str | None) -> str:
    """Return a cache-safe key segment for a login principal."""
    normalized = f"{identifier.strip().lower()}|{ip_address or '-'}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _failure_key(request: HttpRequest, identifier: str) -> str:
    """Return cache key tracking failed authentication attempts."""
    return f"auth-login-fail:{_principal_hash(identifier, client_ip(request))}"


def _lockout_key(request: HttpRequest, identifier: str) -> str:
    """Return cache key tracking an active lockout."""
    return f"auth-login-lock:{_principal_hash(identifier, client_ip(request))}"


def login_is_locked(request: HttpRequest, identifier: str) -> bool:
    """Return whether the principal is temporarily locked out."""
    if not identifier:
        return False
    return bool(cache.get(_lockout_key(request, identifier)))


def record_login_failure(request: HttpRequest, identifier: str) -> bool:
    """
    Record one failed login attempt.

    Returns:
        ``True`` when this attempt activates or hits a lockout.

    """
    if not identifier:
        return False
    limit = _setting_int("AUTH_LOGIN_FAILURE_LIMIT", DEFAULT_LOGIN_FAILURE_LIMIT)
    window = _setting_int("AUTH_LOGIN_WINDOW_SECONDS", DEFAULT_LOGIN_WINDOW_SECONDS)
    lockout = _setting_int("AUTH_LOGIN_LOCKOUT_SECONDS", DEFAULT_LOGIN_LOCKOUT_SECONDS)
    if limit <= 0:
        return False

    failure_key = _failure_key(request, identifier)
    if cache.add(failure_key, 1, timeout=window):
        attempts = 1
    else:
        try:
            attempts = int(cache.incr(failure_key))
        except ValueError:
            cache.set(failure_key, 1, timeout=window)
            attempts = 1

    locked = attempts >= limit
    if locked:
        cache.set(_lockout_key(request, identifier), True, timeout=lockout)
    return locked


def clear_login_failures(request: HttpRequest, identifier: str) -> None:
    """Clear failed login counters for a successfully authenticated principal."""
    if not identifier:
        return
    cache.delete(_failure_key(request, identifier))
    cache.delete(_lockout_key(request, identifier))
