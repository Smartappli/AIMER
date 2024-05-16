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
    """
    Admin class for managing Profile objects in the Django admin interface.

    Attributes:
    - list_display (list): Fields to be displayed in the list view of Profile objects.
    - raw_id_fields (list): Fields for which a raw input field will be provided in the admin interface.

    Example Usage:
    - In the Django admin interface, navigate to the "Profile" section to manage user profiles.
    - The "list_display" specifies the fields to be shown in the list view.
    - The "raw_id_fields" provide a raw input field for the "user" field, facilitating user selection.
    """

    list_display = ["user", "date_of_birth", "photo"]
    raw_id_fields = ["user"]
