from django.contrib import admin
from .models import Model_Family, Model, Model_File, Profile

admin.site.register(Model_Family)
admin.site.register(Model)
admin.site.register(Model_File)


@admin.register(Profile)
class ProfileAdmin(admin.ModelAdmin):
    list_display = ["user", 'date_of_birth', 'photo']
    raw_id_fields = ['user']
