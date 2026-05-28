# Copyright (c) 2026 AIMER contributors.
"""Add security audit events and email-token expiry metadata."""

from typing import ClassVar

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


def clear_legacy_email_tokens(apps, _schema_editor) -> None:
    """Invalidate unhashed one-time tokens from earlier releases."""
    profile = apps.get_model("accounts", "Profile")
    profile.objects.exclude(email_token__isnull=True).update(
        email_token=None,
        email_token_expires_at=None,
    )
    profile.objects.exclude(forget_password_token__isnull=True).update(
        forget_password_token=None,
        forget_password_token_expires_at=None,
    )


class Migration(migrations.Migration):
    """Create security audit storage and invalidate legacy email tokens."""

    dependencies: ClassVar[list] = [
        ("accounts", "0001_initial"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations: ClassVar[list] = [
        migrations.AddField(
            model_name="profile",
            name="email_token_expires_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.RunPython(clear_legacy_email_tokens, migrations.RunPython.noop),
        migrations.CreateModel(
            name="SecurityAuditEvent",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("event_type", models.CharField(db_index=True, max_length=80)),
                ("actor_identifier", models.CharField(blank=True, max_length=255)),
                (
                    "ip_address",
                    models.GenericIPAddressField(blank=True, null=True),
                ),
                ("user_agent", models.TextField(blank=True)),
                ("path", models.CharField(blank=True, max_length=512)),
                ("metadata", models.JSONField(blank=True, default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True, db_index=True)),
                (
                    "user",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="security_audit_events",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "verbose_name": "Security audit event",
                "verbose_name_plural": "Security audit events",
                "ordering": ["-created_at"],
            },
        ),
    ]
