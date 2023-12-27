import random

# import pycaret
from pycaret.datasets import get_data
from pycaret.regression import *

from fl_client.models import Queue
from fl_client.models import Model, Model_File
from fl_client.models import Dataset, Dataset_File
from fl_client.models import Dataset_Central_Data, Dataset_Local_Data, Dataset_Remote_Data

session_seed = random.randrange(1,1000)

tasks = Queue.objects.get(queue_state='CR', queue_model_type='DLCL')
for task in tasks:
    model_id = task.queue_model_id
    dataset_id = task.queue_dataset_id
    params = task.queue_model.params



