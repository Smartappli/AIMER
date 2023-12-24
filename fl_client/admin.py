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
from .models import Model_Family, Model, Model_File, Document, Model_Document, Profile, Queue

admin.site.register(Model_Family)
admin.site.register(Model)
admin.site.register(Model_File)
admin.site.register(Document)
admin.site.register(Model_Document)
admin.site.register(Queue)


@admin.register(Profile)
class ProfileAdmin(admin.ModelAdmin):
    list_display = ["user", 'date_of_birth', 'photo']
    raw_id_fields = ['user']
