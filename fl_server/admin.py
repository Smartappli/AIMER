from django.contrib import admin
from .models import ServerProject
from .models import AgentConfiguration
from .models import ServerModel
from .models import ServerAggregator
from .models import FederatedAuthorisation

admin.site.register(ServerProject)
admin.site.register(AgentConfiguration)
admin.site.register(ServerModel)
admin.site.register(ServerAggregator)
admin.site.register(FederatedAuthorisation)
