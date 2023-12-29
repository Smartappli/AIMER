from django.contrib import admin
from .models import ModelFamily, Model, ModelFile
from .models import Document, ModelDocument
from .models import Profile, Queue, Help, License
from .models import Dataset, DatasetFile
from .models import DatasetLocalData, DatasetRemoteData, DatasetCentralData

admin.site.register(ModelFamily)
admin.site.register(Model)
admin.site.register(ModelFile)
admin.site.register(Document)
admin.site.register(ModelDocument)
admin.site.register(Queue)
admin.site.register(Help)
admin.site.register(Dataset)
admin.site.register(License)
admin.site.register(DatasetFile)
admin.site.register(DatasetLocalData)
admin.site.register(DatasetRemoteData)
admin.site.register(DatasetCentralData)

@admin.register(Profile)
class ProfileAdmin(admin.ModelAdmin):
    """Manage Profile objects"""
    list_display = ["user", 'date_of_birth', 'photo']
    raw_id_fields = ['user']
