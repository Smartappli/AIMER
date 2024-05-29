import secrets

from fl_client.models import Queue

# from fl_client.models import Model, Model_File
# from fl_client.models import Dataset, Dataset_File
# from fl_client.models import Dataset_Central_Data, Dataset_Local_Data, Dataset_Remote_Data

# import pycaret
# from pycaret.datasets import get_data
# from pycaret.classification import *

session_seed = secrets.randbelow(1000) + 1

tasks = Queue.objects.get(queue_state="CR", queue_model_type="MLCL")
for task in tasks:
    model_id = task.queue_model_id
    dataset_id = task.queue_dataset_id
    params = task.queue_model.params
