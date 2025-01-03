# Generated by Django 5.0 on 2023-12-20 21:58

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("fl_client", "0011_alter_model_file_model_file_extension"),
    ]

    operations = [
        migrations.AlterModelOptions(
            name="document",
            options={"ordering": ["document_filename"]},
        ),
        migrations.AlterModelOptions(
            name="model",
            options={"ordering": ["model_name"]},
        ),
        migrations.AlterModelOptions(
            name="model_family",
            options={"ordering": ["model_family_name"]},
        ),
        migrations.AlterField(
            model_name="model",
            name="model_provider",
            field=models.CharField(
                choices=[
                    ("HF", "Hugging Face"),
                    ("KE", "Keras"),
                    ("PC", "PyCaret"),
                    ("PT", "PyTorch"),
                ],
                default="HF",
                max_length=2,
            ),
        ),
    ]
