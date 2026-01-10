from django.contrib import admin
from .models import Profile

# Register your models here.
class Member(admin.ModelAdmin):
    list_display = (
        "user",
        "email",
        "is_verified",
        "created_at",
    )


admin.site.register(Profile, Member)
