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


class Server_Project(models.Model):
    server_project_id = models.BigAutoField(primary_key=True)
    server_project_title = models.CharField(max_length=250)
    server_project_description = models.TextField()
    server_project_owner = models.ForeignKey(User,
                                      on_delete=models.DO_NOTHING,
                                      related_name='server_project_owner')
    server_project_creation_date = models.DateTimeField(auto_now_add=True)
    server_project_updated_date = models.DateTimeField(auto_now=True)



