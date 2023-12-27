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
