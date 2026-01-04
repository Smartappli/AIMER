from django.contrib import admin

from .models import (
    AgentConfiguration,
    FederatedAuthorisation,
    ServerAggregator,
    ServerModel,
    ServerProject,
)

admin.site.register(ServerProject)
admin.site.register(AgentConfiguration)
admin.site.register(ServerModel)
admin.site.register(ServerAggregator)
admin.site.register(FederatedAuthorisation)
