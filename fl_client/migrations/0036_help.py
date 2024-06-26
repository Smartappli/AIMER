# Generated by Django 5.0 on 2023-12-24 12:29

import uuid
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("fl_client", "0035_model_file_model_file_max_ram_required"),
    ]

    operations = [
        migrations.CreateModel(
            name="Help",
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
                (
                    "help_uuid",
                    models.UUIDField(
                        default=uuid.uuid4, editable=False, unique=True
                    ),
                ),
                (
                    "help_key",
                    models.CharField(
                        editable=False, max_length=15, unique=True
                    ),
                ),
                ("help_value", models.CharField(max_length=250)),
            ],
        ),
    ]
