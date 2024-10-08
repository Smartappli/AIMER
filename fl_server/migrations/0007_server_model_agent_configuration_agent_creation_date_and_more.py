# Generated by Django 5.0 on 2023-12-15 10:38

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("fl_server", "0006_alter_agent_configuration_agent_id"),
    ]

    operations = [
        migrations.CreateModel(
            name="Server_Model",
            fields=[
                (
                    "model_id",
                    models.BigAutoField(primary_key=True, serialize=False),
                ),
                ("model_name", models.CharField(max_length=250)),
                ("model_description", models.TextField(blank=True, null=True)),
                (
                    "model_training",
                    models.CharField(
                        choices=[
                            ("FS", "On the Federated Server"),
                            ("LA", "On a Specific Local Agent"),
                            ("AA", "On all Local Participating Agents"),
                            (
                                "AX",
                                "On all Local Participating Agents excluding some of them",
                            ),
                            ("FL", "In Federated Learning"),
                        ],
                        default="LA",
                        max_length=2,
                    ),
                ),
                (
                    "model_origin",
                    models.CharField(
                        choices=[
                            ("AG", "Local Agent"),
                            ("AS", "Aggregator Server"),
                        ],
                        default="AG",
                        max_length=2,
                    ),
                ),
                (
                    "model_creation_date",
                    models.DateTimeField(auto_now_add=True),
                ),
                ("model_updated_date", models.DateTimeField(auto_now=True)),
            ],
        ),
        migrations.AddField(
            model_name="agent_configuration",
            name="agent_creation_date",
            field=models.DateTimeField(auto_now_add=True, null=True),
        ),
        migrations.AddField(
            model_name="agent_configuration",
            name="agent_updated_date",
            field=models.DateTimeField(auto_now=True, null=True),
        ),
        migrations.AlterField(
            model_name="server_project",
            name="server_project_description",
            field=models.TextField(blank=True, null=True),
        ),
        migrations.CreateModel(
            name="Server_Aggregator",
            fields=[
                (
                    "server_aggregator_id",
                    models.BigAutoField(primary_key=True, serialize=False),
                ),
                (
                    "server_aggregator_method",
                    models.CharField(
                        choices=[("FA", "FedAvg")], default="FA", max_length=2
                    ),
                ),
                (
                    "server_aggregator_creation_date",
                    models.DateTimeField(auto_now_add=True),
                ),
                (
                    "server_aggregator_updated_date",
                    models.DateTimeField(auto_now=True),
                ),
                (
                    "server_aggregator_model_id",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.DO_NOTHING,
                        related_name="server_aggregator_model_id",
                        to="fl_server.server_model",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="Federated_Authorisation",
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
                    "federated_autorisation_permission",
                    models.CharField(
                        choices=[("IC", "Included"), ("EC", "Excluded")],
                        default="IC",
                        max_length=2,
                    ),
                ),
                (
                    "federated_autorisation_state",
                    models.CharField(
                        choices=[
                            ("IN", "Invited"),
                            ("RJ", "Rejected"),
                            ("AC", "Accepted"),
                            ("SP", "Suspended"),
                        ],
                        default="IN",
                        max_length=2,
                    ),
                ),
                (
                    "federated_autorisation_creation_date",
                    models.DateTimeField(auto_now_add=True),
                ),
                (
                    "federated_autorisation_updated_date",
                    models.DateTimeField(auto_now=True),
                ),
                (
                    "federated_autorisation_agent_id",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="federated_authorisation_agent_id",
                        to="fl_server.agent_configuration",
                    ),
                ),
                (
                    "federated_autorisation_model_id",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.DO_NOTHING,
                        related_name="federated_authorisation_model_id",
                        to="fl_server.server_model",
                    ),
                ),
            ],
        ),
    ]
