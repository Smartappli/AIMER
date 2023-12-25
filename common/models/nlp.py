from fl_client.models import Queue, Model, Model_File

tasks = Queue.objects.filters(queue_state='CR', queue_model_type='NLTG')
for task in tasks:
    model_id = task.queue_model_id
    params = task.queue_model.params


