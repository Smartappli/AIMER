"""
Copyright (C) 2024  Olivier DEBAUCHE

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from django.contrib.auth.models import User
from django.db import models


class Agent_Configuration(models.Model):
    agent_id = models.BigAutoField(primary_key=True)
    agent_name = models.CharField(max_length=250)
    agent_description = models.CharField(max_length=250)
    agent_creator = models.ForeignKey(User,
                                      on_delete=models.DO_NOTHING,
                                      default=1,
                                      related_name='agent_creator')
    agent_ip = models.GenericIPAddressField(default='127.0.0.1')
    agent_port = models.CharField(max_length=5, default='8765')
    agent_state = models.CharField(max_length=10, default='offline')
    agent_creation_date = models.DateTimeField(auto_now_add=True, null=True)
    agent_updated_date = models.DateTimeField(auto_now=True, null=True)

    def __str__(self):
        return self.agent_name + ' ---- ' + self.agent_state


class Server_Project(models.Model):
    server_project_id = models.BigAutoField(primary_key=True)
    server_project_title = models.CharField(max_length=250)
    server_project_description = models.TextField(null=True, blank=True)
    server_project_owner = models.ForeignKey(User,
                                             on_delete=models.DO_NOTHING,
                                             related_name='server_project_owner')
    server_project_creation_date = models.DateTimeField(auto_now_add=True)
    server_project_updated_date = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.server_project_title


class Server_Model(models.Model):
    class Training(models.TextChoices):
        FS = 'FS', 'On the Federated Server'
        LA = 'LA', 'On a Specific Local Agent'
        AA = 'AA', 'On all Local Participating Agents'
        AX = 'AX', 'On all Local Participating Agents excluding some of them'
        FL = 'FL', 'In Federated Learning'

    class Origin(models.TextChoices):
        AG = 'AG', 'Local Agent'
        AS = 'AS', 'Aggregator Server'

    model_id = models.BigAutoField(primary_key=True)
    model_name = models.CharField(max_length=250)
    model_description = models.TextField(null=True, blank=True)
    model_training = models.CharField(max_length=2,
                                      choices=Training.choices,
                                      default=Training.LA)
    model_origin = models.CharField(max_length=2,
                                    choices=Origin.choices,
                                    default=Origin.AG)
    model_creation_date = models.DateTimeField(auto_now_add=True)
    model_updated_date = models.DateTimeField(auto_now=True)


class Server_Aggregator(models.Model):
    class Method(models.TextChoices):
        FA = 'FA', 'FedAvg'
        FC = 'FC', 'FedCurv'
        FP = 'FP', 'FedProx'
        GM = 'GM', 'Geometric median'
        CM = 'CM', 'Coordinate-wise median'
        KA = 'KA', 'Krum algorithm'

    server_aggregator_id = models.BigAutoField(primary_key=True)
    server_aggregator_model_id = models.ForeignKey(Server_Model,
                                                   on_delete=models.DO_NOTHING,
                                                   related_name='server_aggregator_model_id')
    server_aggregator_method = models.CharField(max_length=2,
                                                choices=Method.choices,
                                                default=Method.FA)
    server_aggregator_creation_date = models.DateTimeField(auto_now_add=True)
    server_aggregator_updated_date = models.DateTimeField(auto_now=True)


class Federated_Authorisation(models.Model):
    class Permission(models.TextChoices):
        IC = 'IC', 'Included'
        EC = 'EC', 'Excluded'

    class State(models.TextChoices):
        IN = 'IN', 'Invited'
        RJ = 'RJ', 'Rejected'
        AC = 'AC', 'Accepted'
        SP = 'SP', 'Suspended'

    federated_autorisation_model_id = models.ForeignKey(Server_Model,
                                                        on_delete=models.DO_NOTHING,
                                                        related_name='federated_authorisation_model_id')
    federated_autorisation_agent_id = models.ForeignKey(Agent_Configuration,
                                                        on_delete=models.CASCADE,
                                                        related_name='federated_authorisation_agent_id')
    federated_autorisation_permission = models.CharField(max_length=2,
                                                         choices=Permission.choices,
                                                         default=Permission.IC)
    federated_autorisation_state = models.CharField(max_length=2,
                                                    choices=State.choices,
                                                    default=State.IN)
    federated_autorisation_creation_date = models.DateTimeField(auto_now_add=True)
    federated_autorisation_updated_date = models.DateTimeField(auto_now=True)
