from django.contrib.auth.models import User
from django.db import models
from django.conf import settings


class Profile(models.Model):
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE)
    date_of_birth = models.DateField(blank=True, null=True)
    photo = models.ImageField(upload_to='users/%Y/%m/%d/', blank=True)

    def __str__(self):
        return f'Profile of {self.user.username}'


class NLP_Model_Family(models.Model):
    model_family_id = models.BigAutoField(primary_key=True)
    model_family_name = models.CharField(max_length=100)
    model_family_owner = models.ForeignKey(User,
                                           on_delete=models.CASCADE,
                                           related_name='model_family_owner')
    model_family_creation_date = models.DateTimeField(auto_created=True)
    model_family_updated_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.model_family_name


class NLP_Model(models.Model):
    class Provider(models.TextChoices):
        HF = 'HF', 'Hugging Face'

    family_model_id = models.ForeignKey(NLP_Model_Family,
                                        on_delete=models.CASCADE,
                                        related_name='family_model_id')
    model_name = models.CharField(max_length=100)
    model_version = models.CharField(max_length=15)
    model_repo = models.CharField(max_length=250)
    model_owner = models.ForeignKey(User,
                                    on_delete=models.CASCADE,
                                    related_name='model_owner')
    model_creation_date = models.DateTimeField(auto_now_add=True)
    model_updated_date = models.DateTimeField(auto_now=True)
    model_provider = models.CharField(max_length=2,
                                      choices=Provider.choices,
                                      default=Provider.HF)

    def __str__(self):
        return self.model_name


class NLP_Model_File(models.Model):
    NLP_model_id = models.ForeignKey(NLP_Model,
                                     on_delete=models.CASCADE,
                                     related_name='NLP_model_id')
    model_file_type = models.CharField(max_length=250)
    model_file_filename = models.CharField(max_length=250)
    model_file_extension = models.CharField(max_length=6, blank=True, null=True)
    model_filesize = models.BigIntegerField(blank=True, null=True)
    model_file_creation_date = models.DateTimeField(auto_now_add=True)
    model_file_updated_date = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.model_file_filename
