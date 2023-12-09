from django.contrib import admin
from .models import NLP_Model_Family, NLP_Model, NLP_Model_File, Profile

admin.site.register(NLP_Model_Family)
admin.site.register(NLP_Model)
admin.site.register(NLP_Model_File)


@admin.register(Profile)
class ProfileAdmin(admin.ModelAdmin):
    list_display = ["user", 'date_of_birth', 'photo']
    raw_id_fields = ['user']
