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

import random

# import pycaret
# from pycaret.datasets import get_data
# from pycaret.classification import *

from fl_client.models import Queue
# from fl_client.models import Model, Model_File
# from fl_client.models import Dataset, Dataset_File
# from fl_client.models import Dataset_Central_Data, Dataset_Local_Data, Dataset_Remote_Data

session_seed = random.randrange(1,1000)

tasks = Queue.objects.get(queue_state='CR', queue_model_type='DLCL')
for task in tasks:
    model_id = task.queue_model_id
    dataset_id = task.queue_dataset_id
    params = task.queue_model.params
