import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
import syft as sy

# Créer un hook PySyft pour étendre PyTorch avec des fonctionnalités de Federated Learning
hook = sy.TorchHook(torch)


# Créer des workers virtuels pour simuler des appareils distants
bob = sy.Worker(hook, id="bob")
alice = sy.Worker(hook, id="alice")

# Charger les données et les diviser entre les travailleurs
# Assurez-vous d'avoir vos propres données et de les charger ici
# Dans cet exemple, nous utilisons le jeu de données CIFAR-10
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# Diviser les données entre les travailleurs
data_bob, data_alice = data_loader.dataset.data.split([len(data_loader.dataset) // 2, len(data_loader.dataset) // 2])

data_bob = transforms.functional.to_tensor(data_bob / 255.0).send(bob)
data_alice = transforms.functional.to_tensor(data_alice / 255.0).send(alice)

# Créer un modèle VGG16
model = models.vgg16(pretrained=False)
model.classifier[6] = nn.Linear(4096, 2)  # Modifier la couche de sortie pour le nombre de classes souhaité

# Définir un optimiseur et une fonction de perte
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()


# Définir la fonction d'entraînement
def train(epoch, model, data, target, optimizer, criterion):
    model.send(data.location)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    model.get()
    return loss.get()


# Entraîner le modèle sur les deux partenaires
for epoch in range(5):  # Vous pouvez ajuster le nombre d'époques en fonction de vos besoins
    # Entraîner sur Bob
    loss_bob = train(epoch, model, data_bob, torch.randint(0, 2, (len(data_bob),)), optimizer, criterion)

    # Entraîner sur Alice
    loss_alice = train(epoch, model, data_alice, torch.randint(0, 2, (len(data_alice),)), optimizer, criterion)

    # Afficher la perte totale après chaque époque
    print(f"Epoch {epoch + 1}, Loss Bob: {loss_bob.item()}, Loss Alice: {loss_alice.item()}")

# Fusionner les modèles de Bob et Alice pour créer un modèle global
model_global = model.fix_precision().share(bob, alice, crypto_provider=sy.Worker(hook, id="crypto"))
