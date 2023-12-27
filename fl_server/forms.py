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
from django import forms
from django.contrib.auth.models import User
from .models import AgentConfiguration


class AgentEditForm(forms.ModelForm):
    """Custom form for editing Agent_Configuration."""
    class Meta:
        model = AgentConfiguration
        fields = ['agent_name', 'agent_description', 'agent_ip', 'agent_port', 'agent_state']
