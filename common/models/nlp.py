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

from huggingface_hub import hf_hub_download, try_to_load_from_cache, _CACHED_NO_EXIST

from fl_client.models import Queue
from fl_client.models import Model, ModelFile
# from fl_client.models import Dataset, Dataset_File
# from fl_client.models import Dataset_Central_Data, Dataset_Local_Data, Dataset_Remote_Data

tasks = Queue.objects.filters(queue_state='CR', queue_model_type='NLTG')
for task in tasks:
    model_id = task.queue_model_id
    params = task.queue_model.params
    dataset_id = task.queue_model_id.dataset_id

    p = Model.objects.get(pk=model_id)
    files = ModelFile.objects.filters(model_file_id=p.model_id)

    for q in files:
        filepath = try_to_load_from_cache(repo_id=p.model_repo,
                                          filename=q.model_file_filename,
                                          repo_type="model")
        if isinstance(filepath, str):
            # file exists and is cached
            print("File in cache")
            print(filepath)
        elif filepath is _CACHED_NO_EXIST:
            # non-existence of file is cached
            print("File in download")
            hf_hub_download(repo_id=p.model_repo, filename=q.model_file_filename)
            print("File downloaded")
        else:
            print("File in download")
            hf_hub_download(repo_id=p.model_repo,
                            filename=q.model_file_filename)
            print("File downloaded")
