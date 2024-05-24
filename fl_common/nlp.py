from huggingface_hub import (
    hf_hub_download,
    try_to_load_from_cache,
    _CACHED_NO_EXIST,
)

from fl_client.models import Queue
from fl_client.models import Model, ModelFile

# from fl_client.models import Dataset, Dataset_File
# from fl_client.models import Dataset_Central_Data, Dataset_Local_Data, Dataset_Remote_Data

tasks = Queue.objects.filters(queue_state="CR", queue_model_type="NLTG")
for task in tasks:
    model_id = task.queue_model_id
    params = task.queue_model.params
    dataset_id = task.queue_model_id.dataset_id

    p = Model.objects.get(pk=model_id)
    files = ModelFile.objects.filters(model_file_id=p.model_id)

    for q in files:
        filepath = try_to_load_from_cache(
            repo_id=p.model_repo,
            filename=q.model_file_filename,
            repo_type="model",
        )
        if isinstance(filepath, str):
            # file exists and is cached
            print("File in cache")
            print(filepath)
        elif filepath is _CACHED_NO_EXIST:
            # non-existence of file is cached
            print("File in download")
            hf_hub_download(
                repo_id=p.model_repo, filename=q.model_file_filename
            )
            print("File downloaded")
        else:
            print("File in download")
            hf_hub_download(
                repo_id=p.model_repo, filename=q.model_file_filename
            )
            print("File downloaded")
