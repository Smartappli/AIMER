# Generated by Django 5.0 on 2023-12-24 12:34

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("fl_client", "0037_alter_help_help_key"),
    ]

    operations = [
        migrations.AlterModelOptions(
            name="help",
            options={"ordering": ["help_key"]},
        ),
    ]
