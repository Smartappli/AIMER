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

import uuid

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
    local_project_uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    local_project_title = models.CharField(max_length=250)
    local_project_description = models.TextField()
    local_project_type = models.CharField(max_length=2,
                                          choices=Project_Type.choices,
                                          default=Project_Type.LC)
    local_project_active = models.BooleanField(default=True)
    local_project_owner = models.ForeignKey(User,
                                            on_delete=models.CASCADE,
                                            related_name='local_project_owner')
    local_project_creation_date = models.DateTimeField(auto_now_add=True)
    local_project_updated_date = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.local_project_title


class License(models.Model):
    license_id = models.BigAutoField(primary_key=True, editable=False)
    license_uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    license_short_name = models.CharField(max_length=30, null=True, blank=True)
    license_name = models.CharField(max_length=250)
    license_description = models.TextField(blank=True)
    license_owner = models.ForeignKey(User,
                                      on_delete=models.DO_NOTHING,
                                      default=1,
                                      related_name='license_owner')
    license_creation_date = models.DateTimeField(auto_now_add=True)
    license_updated_date = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['license_short_name']

    def __str__(self):
        return (self.license_short_name
                + ' - '
                + self.license_name)


# ---- Model tables ----
class Model_Family(models.Model):
    model_family_id = models.BigAutoField(primary_key=True)
    model_family_uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    model_family_name = models.CharField(max_length=100)
    model_family_active = models.BooleanField(default=True)
    model_family_owner = models.ForeignKey(User,
                                           on_delete=models.CASCADE,
                                           related_name='model_family_owner')
    model_family_creation_date = models.DateTimeField(auto_now_add=True)
    model_family_updated_date = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['model_family_name']

    def __str__(self):
        return self.model_family_name


class Model(models.Model):
    class Provider(models.TextChoices):
        HF = 'HF', 'Hugging Face'
        KE = 'KE', 'Keras'
        PC = 'PC', 'PyCaret'
        PT = 'PT', 'PyTorch'

    class Category(models.TextChoices):
        DL = 'DL', 'Deep Learning'
        ML = 'ML', 'Machine Learning'
        NL = 'NL', 'Natural Language Processing'

    class Type(models.TextChoices):
        AD = 'AD', 'Anomaly Detection'
        CL = 'CL', 'Classification'
        CU = 'CU', 'Clustering'
        RG = 'RG', 'Regression'
        SG = 'SG', 'Segmentation'
        TC = 'TC', 'Text-Classification'
        TG = 'TG', 'Text-Generation'
        TS = 'TS', 'Time-Series'

    model_id = models.BigAutoField(primary_key=True, editable=False)
    model_uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    model_name = models.CharField(max_length=200)
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
    model_provider = models.CharField(max_length=2,
                                      choices=Provider.choices,
                                      default=Provider.HF)
    model_repo = models.CharField(max_length=250, null=True, blank=True)
    model_license = models.ForeignKey(License,
                                      on_delete=models.DO_NOTHING,
                                      default=1,
                                      related_name='model_license')
    model_active = models.BooleanField(default=True)
    model_owner = models.ForeignKey(User,
                                    on_delete=models.DO_NOTHING,
                                    related_name='model_owner')
    model_creation_date = models.DateTimeField(auto_now_add=True)
    model_updated_date = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['model_name']

    def __str__(self):
        if str(self.model_version) != "None":
            return (self.model_category
                    + self.model_type
                    + ' - '
                    + self.model_name
                    + ' - v'
                    + str(self.model_version))
        else:
            return (self.model_category
                    + self.model_type
                    + ' - '
                    + self.model_name)


class Model_File(models.Model):
    class Type(models.TextChoices):
        NONE = 'NA', 'N/A'
        Q2K = 'Q2K', 'Q2_K'
        Q3KL = 'Q3KL', 'Q3_K_L'
        Q3KM = 'Q3KM', 'Q3_K_M'
        Q3KS = 'Q3KS', 'Q3_K_S'
        Q40 = 'Q40', 'Q4_0'
        Q41 = 'Q41', 'Q4_1'
        Q4KM = 'Q4KM', 'Q4_K_M'
        Q4KS = 'Q4KS', 'Q4_K_S'
        Q50 = 'Q50', 'Q5_0'
        Q51 = 'Q51', 'Q5_1'
        Q5KM = 'Q5KM', 'Q5_K_M'
        Q5KS = 'Q5KS', 'Q5_K_S'
        Q6K = 'Q6K', 'Q6_K'
        Q80 = 'Q80', 'Q8_0'

    class Extension(models.TextChoices):
        NONE = 'NA', 'N/A'
        BIN = 'BIN', 'Binary'
        GGUF = 'GGUF', 'GGUF'

    model_file_uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    model_file_model_id = models.ForeignKey(Model,
                                            on_delete=models.CASCADE,
                                            default=1,
                                            related_name='model_file_model_id')
    model_file_type = models.CharField(max_length=4,
                                       choices=Type.choices,
                                       default=Type.NONE)
    model_file_filename = models.CharField(max_length=250, unique=True)
    model_file_extension = models.CharField(max_length=6,
                                            choices=Extension.choices,
                                            default=Extension.NONE)
    model_file_size = models.BigIntegerField(blank=True, null=True)
    model_file_sha256 = models.CharField(max_length=64, blank=True, null=True)
    model_file_creation_date = models.DateTimeField(auto_now_add=True)
    model_file_updated_date = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['model_file_filename']

    def __str__(self):
        return (self.model_file_model_id.model_name
                + ' | '
                + self.model_file_type
                + " --- "
                + self.model_file_extension
                + ' --- '
                + self.model_file_filename)


class Document(models.Model):
    document_model_id = models.BigAutoField(primary_key=True, editable=False)
    document_uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    document_title = models.CharField(max_length=250)
    document_filename = models.CharField(max_length=250, default="")
    document_active = models.BooleanField(default=True)
    document_owner = models.ForeignKey(User,
                                       on_delete=models.DO_NOTHING,
                                       default=1,
                                       related_name='document_owner')
    document_creation_date = models.DateTimeField(auto_now_add=True)
    document_updated_date = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['document_filename']

    def __str__(self):
        return (self.document_filename
                + ' ----- '
                + self.document_title)


class Model_Document(models.Model):
    modeldoc_uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    modeldoc_model_id = models.ForeignKey(Model,
                                          on_delete=models.CASCADE,
                                          default=1,
                                          related_name='modeldoc_model_id')
    modeldoc_document = models.ForeignKey(Document,
                                          on_delete=models.CASCADE,
                                          default=1,
                                          related_name='modeldoc_document_id')
    modeldoc_active = models.BooleanField(default=True)
    modeldoc_owner = models.ForeignKey(User,
                                       on_delete=models.DO_NOTHING,
                                       default=1,
                                       related_name='modeldoc_owner')
    modeldoc_creation_date = models.DateTimeField(auto_now_add=True, blank=True, null=True)
    modeldoc_updated_date = models.DateTimeField(auto_now=True, blank=True, null=True)

    class Meta:
        ordering = ['modeldoc_model_id']

    def __str__(self):
        return (self.modeldoc_model_id.model_name
                + ' ----- '
                + self.modeldoc_document.document_filename
                + '  |  '
                + self.modeldoc_document.document_title)


class Dataset(models.Model):
    class Format(models.TextChoices):
        CSV = 'CSV', 'Comma-separated values'
        DICOM = 'DICOM', 'DICOM'
        FHIR = 'FHIR', 'FHIR'
        SNOMED = 'SNOMED', 'SNOMED CT'
        IMG = 'IMG', 'Images'
        JSON = 'JSON', 'JavaScript Object Notation'
        TXT = 'TXT', 'Text'

    class Type(models.TextChoices):
        LC = 'LC', 'Lolly hosted'
        CS = 'CS', 'On the central server'
        EH = 'EH', 'Externally Hosted'

    dataset_id = models.BigAutoField(primary_key=True, editable=False)
    dataset_uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    dataset_name = models.CharField(max_length=250)
    dataset_description = models.TextField(blank=True)
    dataset_type = models.CharField(max_length=6,
                                    choices=Type.choices,
                                    default=Type.LC)
    dataset_format = models.CharField(max_length=6,
                                      choices=Format.choices,
                                      default=Format.CSV)
    dataset_licence = models.ForeignKey(License,
                                        on_delete=models.DO_NOTHING,
                                        default=1,
                                        related_name='dataset_license')
    dataset_owner = models.ForeignKey(User,
                                      on_delete=models.DO_NOTHING,
                                      default=1,
                                      related_name='dataset_owner')
    dataset_creation_date = models.DateTimeField(auto_now_add=True)
    dataset_updated_date = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['dataset_name']

    def __str__(self):
        return self.dataset_name


class Dataset_Local_Data(models.Model):
    dataset_local_data_id = models.BigAutoField(primary_key=True, editable=False)
    dataset_local_data_uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    dataset_local_data_dataset_id = models.ForeignKey(Dataset,
                                                      on_delete=models.DO_NOTHING,
                                                      default=1,
                                                      related_name='ds_local_data_dataset_id')
    dataset_local_data_link = models.URLField(max_length=255)
    dataset_local_data_username = models.CharField(max_length=30, blank=True, null=True)
    dataset_local_data_password = models.CharField(max_length=30, blank=True, null=True)
    dataset_local_data_creation_date = models.DateTimeField(auto_now_add=True)
    dataset_local_data_updated_date = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['dataset_local_data_link']

    def __str__(self):
        return (self.dataset_local_data_dataset_id.dataset_name
                + ' - '
                + self.dataset_local_data_link)


class Dataset_Remote_Data(models.Model):
    class Protocol(models.TextChoices):
        HTTP = 'HTTP', 'HTTP'
        HTTPS = 'HTTPS', 'HTTPS'
        FTP = 'FTP', 'FTP'
        FTPS = 'FTPS', 'FTPS'
        SCP = 'SCP', 'SCP'
        SFTP = 'SFTP', 'SFTP'

    dataset_remote_data_id = models.BigAutoField(primary_key=True, editable=False)
    dataset_remote_data_uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    dataset_remote_data_dataset_id = models.ForeignKey(Dataset,
                                                       on_delete=models.DO_NOTHING,
                                                       default=1,
                                                       related_name='ds_remote_data_dataset_id')
    dataset_remote_data_protocol = models.CharField(max_length=6,
                                                    choices=Protocol.choices,
                                                    default=Protocol.HTTP)
    dataset_remote_data_username = models.CharField(max_length=30, blank=True, null=True)
    dataset_remote_data_password = models.CharField(max_length=30, blank=True, null=True)
    dataset_remote_data_ip = models.CharField(max_length=40, blank=True, null=True)
    dataset_remote_data_path = models.CharField(max_length=255, blank=True, null=True)
    dataset_remote_creation_date = models.DateTimeField(auto_now_add=True)
    dataset_remote_updated_date = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['dataset_remote_data_path']

    def __str__(self):
        return (self.dataset_remote_data_dataset_id.dataset_name
                + ' - '
                + str(self.dataset_remote_data_ip)
                + ' - '
                + self.dataset_remote_data_path)


class Dataset_Central_Data(models.Model):
    dataset_central_data_id = models.BigAutoField(primary_key=True, editable=False)
    dataset_central_data_uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    dataset_central_data_dataset_id = models.ForeignKey(Dataset,
                                                        on_delete=models.DO_NOTHING,
                                                        default=1,
                                                        related_name='ds_central_data_dataset_id')
    dataset_central_data_link = models.CharField(max_length=255, blank=True)
    dataset_central_creation_date = models.DateTimeField(auto_now_add=True)
    dataset_central_updated_date = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['dataset_central_data_link']

    def __str__(self):
        return (self.dataset_central_data_dataset_id.dataset_name
                + ' - '
                + str(self.dataset_central_data_link))


class Dataset_File(models.Model):
    dataset_file_id = models.BigAutoField(primary_key=True, editable=False)
    dataset_file_uuid = models.UUIDField(default=uuid.uuid4, editable=False)
    dataset_file_dataset_id = models.ForeignKey(Dataset,
                                                on_delete=models.DO_NOTHING,
                                                default=1,
                                                related_name='dataset_file_dataset_id')
    dataset_file_filename = models.CharField(max_length=255)
    dataset_file_creation_date = models.DateTimeField(auto_now_add=True)
    dataset_file_updated_date = models.DateTimeField(auto_now=True)


# --- Processing ----
class Queue(models.Model):
    class State(models.TextChoices):
        CL = 'CL', 'Cancelled'
        CP = 'CP', 'Completed'
        CR = 'CR', 'Created'
        ER = 'ER', 'Error'
        IP = 'IP', 'In progress'
        PN = 'PN', 'Pending'
        RL = 'RL', 'Rejected'
        RS = 'RS', 'Restarted'
        ST = 'ST', 'Started'
        SP = 'SP', 'Stopped'
        UP = 'UP', 'Updated'

    queue_id = models.BigAutoField(primary_key=True, default=1, editable=False)
    queue_uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    queue_model_id = models.ForeignKey(Model,
                                       on_delete=models.CASCADE,
                                       related_name='queue_model_id')
    queue_model_type = models.CharField(max_length=4,
                                        null=True,
                                        blank=True)
    queue_params = models.JSONField(default=dict)
    queue_dataset_id = models.ForeignKey(Dataset,
                                         default=1,
                                         on_delete=models.CASCADE,
                                         related_name='queue_dataset_id')
    queue_state = models.CharField(max_length=2,
                                   choices=State.choices,
                                   default=State.CR)
    queue_owner = models.ForeignKey(User,
                                    on_delete=models.CASCADE,
                                    default=1,
                                    related_name='queue_owner')
    queue_creation_date = models.DateTimeField(auto_now_add=True)
    queue_updated_date = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['queue_model_type', 'queue_state']

    def __str__(self):
        return str(self.queue_uuid)


class Help(models.Model):
    help_uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    help_key = models.CharField(max_length=15, unique=True)
    help_value = models.CharField(max_length=255)
    help_creation_date = models.DateTimeField(auto_now_add=True, blank=True, null=True)
    help_updated_date = models.DateTimeField(auto_now=True, blank=True, null=True)

    class Meta:
        ordering = ['help_key']

    def __str__(self):
        return (self.help_key
                + ' : '
                + self.help_value)
