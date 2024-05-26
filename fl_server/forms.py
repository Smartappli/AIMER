from django import forms
from django.contrib.auth.models import User
from .models import AgentConfiguration
from typing import ClassVar


class AgentEditForm(forms.ModelForm):
    """Custom form for editing Agent_Configuration."""

    class Meta:
        model = AgentConfiguration
        fields: ClassVar[list[str]] = [
            "agent_name",
            "agent_description",
            "agent_ip",
            "agent_port",
            "agent_state",
        ]
