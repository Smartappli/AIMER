from typing import ClassVar

from django import forms

from .models import AgentConfiguration


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
