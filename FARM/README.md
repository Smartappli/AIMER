# FARM - Federated Learning

Ce module fournit une base **Flower + Syft** pour orchestrer l'entraînement fédéré de
modèles de **classification**, **détection**, **segmentation** et un **RAG partagé**.

## Structure

- `FARM/federated/clients.py` : clients Flower pour tâches ML et RAG.
- `FARM/federated/strategies.py` : stratégies FedAvg pour ML et agrégation RAG.
- `FARM/federated/tasks.py` : abstractions de tâches (handlers, résultats, types).
- `FARM/federated/rag.py` : index RAG en mémoire, sérialisation Flower.
- `FARM/federated/runner.py` : helpers pour lancer serveurs/clients.

## Exemple (classification/détection/segmentation)

```python
import numpy as np
import flwr as fl

from FARM.federated import (
    FederatedDataset,
    TaskDefinition,
    TaskHandlers,
    TaskType,
    TrainingResult,
    EvaluationResult,
    build_task_client,
)

class SimpleHandlers:
    def get_parameters(self, model):
        return [model["weights"]]

    def set_parameters(self, model, parameters):
        model["weights"] = parameters[0]

    def train(self, model, dataset, config):
        # Exemple minimal: un pas de descente de gradient fictif
        model["weights"] = model["weights"] + 0.1
        return TrainingResult(parameters=[model["weights"]], num_examples=100, metrics={"loss": 0.2})

    def evaluate(self, model, dataset, config):
        return EvaluationResult(loss=0.1, num_examples=50, metrics={"acc": 0.9})

model = {"weights": np.zeros((10,), dtype=np.float32)}
handlers = SimpleHandlers()

dataset = FederatedDataset(train="train_data", validation="val_data")

task = TaskDefinition(
    name="image_classification",
    task_type=TaskType.CLASSIFICATION,
    model=model,
    dataset=dataset,
    handlers=handlers,
)

client = build_task_client(task)
fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)
```

## Exemple (RAG partagé)

```python
import flwr as fl
import numpy as np

from FARM.federated import RagDocument, RagIndex, RagState, build_rag_client

index = RagIndex(embedding_dim=3)

# Le client décide quels documents partager à chaque round

def provide_docs(config):
    docs = [
        RagDocument(doc_id="doc-1", embedding=[0.1, 0.2, 0.3], text="hello", metadata={"lang": "en"})
    ]
    embeddings = [d.embedding for d in docs]
    state = RagState(
        doc_ids=[d.doc_id for d in docs],
        embeddings=np.asarray(embeddings, dtype=np.float32),
        texts=[d.text for d in docs],
        metadata=[d.metadata for d in docs],
    )
    return [state]

client = build_rag_client(index=index, document_provider=provide_docs)
fl.client.start_client(server_address="0.0.0.0:8081", client=client)
```

## Syft

La dépendance `syft-flwr` permet d'utiliser le transport sécurisé de Syft avec Flower.
Les abstractions ci-dessus restent compatibles: vous pouvez remplacer `fl.client.start_*`
par les exécutables Syft (clients/serveurs) afin d'orchestrer des entraînements sur
plusieurs organisations tout en conservant la logique de tâches.
