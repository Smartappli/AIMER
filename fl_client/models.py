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
from django.contrib.auth.models import User
from django.conf import settings
from django.db import models


class Profile(models.Model):
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE)
    date_of_birth = models.DateField(blank=True, null=True)
    photo = models.ImageField(upload_to='users/%Y/%m/%d/', blank=True)

    def __str__(self):
        return f'Profile of {self.user.name}'


# ---- Project tables ----
class Local_Project(models.Model):
    class Project_Type(models.TextChoices):
        LC = 'LC', 'Local Project'
        RM = 'RM', 'Remote Project'
        FD = 'FD', 'Federated Project'

    local_project_id = models.BigAutoField(primary_key=True, default=1)
    local_project_title = models.CharField(max_length=250)
    local_project_description = models.TextField()
    local_project_type = models.CharField(max_length=2,
                                          choices=Project_Type.choices,
                                          default=Project_Type.LC)
    local_project_owner = models.ForeignKey(User,
                                            on_delete=models.CASCADE,
                                            related_name='local_project_owner')
    local_project_creation_date = models.DateTimeField(auto_now_add=True)
    local_project_updated_date = models.DateTimeField(auto_now=True)


# ---- Model tables ----
class Model_Family(models.Model):
    model_family_id = models.BigAutoField(primary_key=True)
    model_family_name = models.CharField(max_length=100)
    model_family_owner = models.ForeignKey(User,
                                           on_delete=models.CASCADE,
                                           related_name='model_family_owner')
    model_family_creation_date = models.DateTimeField(auto_now_add=True)
    model_family_updated_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.model_family_name


class Model(models.Model):
    class Provider(models.TextChoices):
        HF = 'HF', 'Hugging Face',
        KE = 'KE', 'Keras',
        PT = 'PT', 'PyTorch'

    class Category(models.TextChoices):
        DL = 'DL', 'Deep Learning',
        ML = 'ML', 'Machine Learning',
        NL = 'NL', 'Natural Language Processing'

    class Type(models.TextChoices):
        AD = 'AD', 'Anomaly Detection',
        CL = 'CL', 'Classification',
        SG = 'SG', 'Segmentation',
        TC = 'TC', 'Text-Classification',
        TG = 'TG', 'Text-Generation'

    model_id = models.BigAutoField(primary_key=True, default=1, editable=False)
    model_name = models.CharField(max_length=100)
    model_description = models.TextField(null=True, blank=True)
    model_version = models.CharField(max_length=15, null=True, blank=True)
    model_category = models.CharField(max_length=2,
                                      choices=Category.choices,
                                      default=Category.ML)
    model_type = models.CharField(max_length=2,
                                  choices=Type.choices,
                                  default=Type.AD)
    model_family = models.ForeignKey(Model_Family,
                                     on_delete=models.CASCADE,
                                     related_name='family_model')
    model_repo = models.CharField(max_length=250),
    model_owner = models.ForeignKey(User,
                                    on_delete=models.DO_NOTHING,
                                    related_name='model_owner')
    model_creation_date = models.DateTimeField(auto_now_add=True)
    model_updated_date = models.DateTimeField(auto_now=True)
    model_provider = models.CharField(max_length=2,
                                      choices=Provider.choices,
                                      default=Provider.HF)

    def __str__(self):
        return self.model_name


class Model_File(models.Model):
    class Type(models.TextChoices):
        NONE = 'NA', 'N/A',
        Q2K = 'Q2K', 'Q2_K',
        Q3KL = 'Q3KL', 'Q3_K_L',
        Q3KM = 'Q3KM', 'Q3_K_M',
        Q3KS = 'Q3KS', 'Q3_K_S'
        Q40 = 'Q40', 'Q4_0',
        Q4KM = 'Q4KM', 'Q4_K_M',
        Q4KS = 'Q4KS', 'Q4_K_S',
        Q50 = 'Q50', 'Q5_0',
        Q5KM = 'Q5KM', 'Q5_K_M',
        Q5KS = 'Q5KS', 'Q5_K_S',
        Q6K = 'Q6K', 'Q6_K',
        Q80 = 'Q80', 'Q8_0'

    class Extension(models.TextChoices):
        NONE = 'NA', 'N/A',
        BIN = 'BIN', 'Binary',
        GGUF = 'GGUF', 'GGUF'

    model_file_model_id = models.ForeignKey(Model,
                                            on_delete=models.CASCADE,
                                            default=1,
                                            related_name='model_file_model_id')
    model_file_type = models.CharField(max_length=4,
                                       choices=Type.choices,
                                       default=Type.NONE)
    model_file_filename = models.CharField(max_length=250)
    model_file_extension = models.CharField(max_length=6, blank=True, null=True)
    model_filesize = models.BigIntegerField(blank=True, null=True)
    model_file_creation_date = models.DateTimeField(auto_now_add=True)
    model_file_updated_date = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.model_file_filename


class Document(models.Model):
    document_model_id = models.BigAutoField(primary_key=True, editable=False)
    document_title = models.CharField(max_length=250)
    document_filename = models.CharField(max_length=250)
    document_owner = models.ForeignKey(User,
                                       on_delete=models.DO_NOTHING,
                                       default=1,
                                       related_name='model_doc_owner')
    document_creation_date = models.DateTimeField(auto_now_add=True)
    document_updated_date = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.document_filename + ' ----- ' + self.document_title


class Model_Document(models.Model):
    modeldoc_model_id = models.ForeignKey(Model,
                                          on_delete=models.CASCADE,
                                          default=1,
                                          related_name='modeldoc_model_id')
    modeldoc_document = models.ForeignKey(Document,
                                          on_delete=models.CASCADE,
                                          default=1,
                                          related_name='modeldoc_document_id')


# --- Processing ----
class Queue(models.Model):
    class State(models.TextChoices):
        CL = 'CL', 'Cancelled',
        CP = 'CP', 'Completed',
        CR = 'CR', 'Created',
        ER = 'ER', 'Error',
        IP = 'IP', 'In progress'
        PN = 'PN', 'Pending',
        RL = 'RL', 'Rejected',
        RS = 'RS', 'Restarted'
        ST = 'ST', 'Started',
        SP = 'SP', 'Stopped',
        UP = 'UP', 'Updated'

    queue_id = models.BigAutoField(primary_key=True, default=1, editable=False)
    queue_model_id = models.ForeignKey(Model,
                                       on_delete=models.CASCADE,
                                       related_name='queue_model_id')
    queue_state = models.CharField(max_length=2,
                                   choices=State.choices,
                                   default=State.CR)
    queue_owner = models.ForeignKey(User,
                                    on_delete=models.CASCADE,
                                    default=1,
                                    related_name='queue_owner')
    queue_creation_date = models.DateTimeField(auto_now_add=True)
    queue_updated_date = models.DateTimeField(auto_now=True)
