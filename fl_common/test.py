import torch

# Liste des modèles disponibles dans torch.hub
models = torch.hub.list("pytorch/vision")

# Afficher les noms des modèles
print("Modèles disponibles dans torch.hub :")
for model in models:
    print(model)