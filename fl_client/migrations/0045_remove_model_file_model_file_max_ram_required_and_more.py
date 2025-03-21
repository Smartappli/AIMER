# Generated by Django 5.0 on 2023-12-25 01:08

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        (
            "fl_client",
            "0044_alter_model_document_modeldoc_creation_date_and_more",
        ),
    ]

    operations = [
        migrations.RemoveField(
            model_name="model_file",
            name="model_file_max_ram_required",
        ),
        migrations.AddField(
            model_name="model",
            name="model_license",
            field=models.ForeignKey(
                default=1,
                on_delete=django.db.models.deletion.DO_NOTHING,
                related_name="model_license",
                to="fl_client.license",
            ),
        ),
    ]
