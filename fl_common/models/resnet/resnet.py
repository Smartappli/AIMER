from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
import time

# Parameters
dataset_path = 'c:/IA/Data'  # Replace with the actual path to your dataset
resnet_type = 'ResNet50'
optimizer_name = 'Adam'
learning_rate = 0.001
criterion_name = 'CrossEntropyLoss'
batch_size = 32
num_epochs = 100
patience = 7


def get_resnet_model(resnet_type='DenseNet121', num_classes=1000):
    # Load the pre-trained version of DenseNet
    if resnet_type == 'ResNet18':
        weights = models.ResNet18_Weights.DEFAULT
        resnet_model = models.resnet18(weights=weights)
    elif resnet_type == 'ResNet34':
        weights = models.ResNet34_Weights.DEFAULT
        resnet_model = models.resnet34(weights=weights)
    elif resnet_type == 'ResNet50':
        weights = models.ResNet50_Weights.DEFAULT
        resnet_model = models.resnet50(weights=weights)
    elif resnet_type == 'ResNet101':
        weights = models.ResNet101_Weights.DEFAULT
        resnet_model = models.resnet101(weights=weights)
    elif resnet_type == 'ResNet152':
        weights = models.ResNet152_Weights.DEFAULT
        resnet_model = models.resnet152(weights=weights)
    else:
        raise ValueError(f'Unknown DenseNet Architecture : {resnet_type}')

    # Modify last layer to suit number of classes
    num_features = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_features, num_classes)

    return resnet_model


def get_criterion(criterion_name):
    if criterion_name == 'MSELoss':
        return nn.MSELoss()
    elif criterion_name == 'L1Loss':
        return nn.L1Loss()
    elif criterion_name == 'CTCLoss':
        return nn.CTCLoss()
    elif criterion_name == 'KLDivLoss':
        return nn.KLDivLoss()
    elif criterion_name == 'GaussianNLLLoss':
        return nn.GaussianNLLLoss()
    elif criterion_name == 'SmoothL1Loss':
        return nn.SmoothL1Loss()
    elif criterion_name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    elif criterion_name == 'BCELoss':
        return nn.BCELoss()
    elif criterion_name == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss()
    elif criterion_name == 'NLLLoss':
        return nn.NLLLoss()
    elif criterion_name == 'PoissonNLLLoss':
        return nn.PoissonNLLLoss()
    elif criterion_name == 'KLDivLoss':
        return nn.KLDivLoss()
    elif criterion_name == 'MarginRankingLoss':
        return nn.MarginRankingLoss()
    elif criterion_name == 'HingeEmbeddingLoss':
        return nn.HingeEmbeddingLoss()
    elif criterion_name == 'MultiLabelMarginLoss':
        return nn.MultiLabelMarginLoss()
    elif criterion_name == 'SmoothL1Loss':
        return nn.SmoothL1Loss()
    elif criterion_name == 'HuberLoss':
        return nn.HuberLoss()
    elif criterion_name == 'SoftMarginLoss':
        return nn.SoftMarginLoss()
    elif criterion_name == 'MultiLabelSoftMarginLoss':
        return nn.MultiLabelSoftMarginLoss()
    elif criterion_name == 'CosineEmbeddingLoss':
        return nn.CosineEmbeddingLoss()
    elif criterion_name == 'MultiMarginLoss':
        return nn.MultiMarginLoss()
    elif criterion_name == 'TripletMarginLoss':
        return nn.TripletMarginLoss()
    elif criterion_name == 'TripletMarginWithDistanceLoss':
        return nn.TripletMarginWithDistanceLoss()
    else:
        raise ValueError(f'Unknown Criterion : {criterion_name}')


def get_optimizer(optimizer_name, model_parameters, learning_rate):
    if optimizer_name == 'SGD':
        return optim.SGD(model_parameters, lr=learning_rate)
    elif optimizer_name == 'Adam':
        return optim.Adam(model_parameters, lr=learning_rate)
    elif optimizer_name == 'RMSprop':
        return optim.RMSprop(model_parameters, lr=learning_rate)
    elif optimizer_name == 'Adagrad':
        return optim.Adagrad(model_parameters, lr=learning_rate)
    elif optimizer_name == 'Adadelta':
        return optim.Adadelta(model_parameters, lr=learning_rate)
    elif optimizer_name == 'AdamW':
        return optim.AdamW(model_parameters, lr=learning_rate)
    elif optimizer_name == 'SparseAdam':
        return optim.SparseAdam(model_parameters, lr=learning_rate)
    elif optimizer_name == 'Adamax':
        return optim.Adamax(model_parameters, lr=learning_rate)
    elif optimizer_name == 'ASGD':
        return optim.ASGD(model_parameters, lr=learning_rate)
    elif optimizer_name == 'LBFGS':
        return optim.LBFGS(model_parameters, lr=learning_rate)
    elif optimizer_name == 'Rprop':
        return optim.Rprop(model_parameters, lr=learning_rate)
    elif optimizer_name == 'NAdam':
        return optim.NAdam(model_parameters, lr=learning_rate)
    elif optimizer_name == 'RAdam':
        return optim.RAdam(model_parameters, lr=learning_rate)
    else:
        raise ValueError(f'Unknown Optimizer : {optimizer_name}')


def get_scheduler(optimizer, scheduler_type='step', **kwargs):
    """
    Get a learning rate scheduler for the optimizer.

    Parameters:
        optimizer (torch.optim.Optimizer): Optimizer for which the scheduler is desired.
        scheduler_type (str): Type of scheduler to use. Options: 'step', 'multi_step', 'exponential'.
        **kwargs: Additional arguments for the specific scheduler type.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: Learning rate scheduler.
    """
    if scheduler_type == 'step':
        # Example: StepLR
        return lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_type == 'multi_step':
        # Example: MultiStepLR
        return lr_scheduler.MultiStepLR(optimizer, **kwargs)
    elif scheduler_type == 'exponential':
        # Example: ExponentialLR
        return lr_scheduler.ExponentialLR(optimizer, **kwargs)
    else:
        raise ValueError(f"Invalid scheduler_type: {scheduler_type}")


# Define the transformation for the dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load your custom dataset
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Split the dataset into training and testing sets
train_indices, test_indices = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

# Define data loaders
train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

# Use the pre-trained VGG model
num_classes = len(dataset.classes)
model = get_resnet_model(resnet_type=resnet_type, num_classes=num_classes)

# model.classifier[6] = nn.Linear(4096, num_classes)

# Move the model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model = model.to(device)

# Define the loss criterion and optimizer
model_parameters = model.parameters()
criterion = get_criterion(criterion_name)
optimizer = get_optimizer(optimizer_name, model_parameters, learning_rate)

# scheduler = get_scheduler(optimizer, scheduler_type='step', step_size=10, gamma=0.5)
# scheduler = get_scheduler(optimizer, scheduler_type='multi_step', milestones=[30, 60, 90], gamma=0.5)
# scheduler = get_scheduler(optimizer, scheduler_type='exponential', gamma=0.95)

scheduler = get_scheduler(optimizer, scheduler_type='step', step_size=10, gamma=0.5)

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_score = float('inf')
        self.early_stop = False
        self.best_model_weights = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_score:
            self.best_score = val_loss
            self.counter = 0
            self.best_model_weights = model.state_dict()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# Training loop
early_stopping = EarlyStopping(patience=5)

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
elapsed_times = []

start_time = time.time()

for epoch in range(num_epochs):
    epoch_start_time = time.time()

    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    print(f"\nEpoch {epoch + 1}/{num_epochs}, Learning Rate: {optimizer.param_groups[0]['lr']}")

    # Use tqdm for a progress bar over the training batches
    with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}") as progress_bar:
        for batch_idx, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Print progress within the epoch
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                avg_batch_loss = running_loss / (batch_idx + 1)
                batch_accuracy = correct_train / total_train
                progress_bar.set_postfix(Batch_Loss=f"{avg_batch_loss:.4f}", Batch_Accuracy=f"{batch_accuracy:.4f}")

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    train_accuracy = correct_train / total_train
    train_accuracies.append(train_accuracy)

    # Validation
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)
        val_accuracy = correct_val / total_val
        val_accuracies.append(val_accuracy)

    # Update learning rate after the optimizer step
    scheduler.step()

    # Early stopping check
    early_stopping(avg_val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping after {} epochs".format(epoch + 1))
        break

    epoch_end_time = time.time()
    elapsed_time = epoch_end_time - epoch_start_time
    elapsed_times.append(elapsed_time)

    # Print and plot results
    print(f"Epoch {epoch + 1}/{num_epochs} => "
          f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
          f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, "
          f"Elapsed Time: {elapsed_time:.2f} seconds")

# Calculate total training time
end_time = time.time()
total_training_time = end_time - start_time
print(f"\nTotal Training Time: {total_training_time:.2f} seconds")

# Save the best model weights
model.load_state_dict(early_stopping.best_model_weights)
torch.save(model.state_dict(), 'best_model.pth')

# Plot training and validation losses
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.show()

# Plot training and validation accuracies
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracies')
plt.show()

# Performance evaluation
model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Confusion matrix and classification report
conf_matrix = confusion_matrix(all_labels, all_predictions)
class_report = classification_report(all_labels, all_predictions, target_names=dataset.classes)

print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Plot graphical confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=dataset.classes)
disp.plot(cmap='viridis', values_format='d')
plt.title('Confusion Matrix')
plt.show()

# Print final accuracy and validation accuracy
final_accuracy = correct_train / total_train
final_val_accuracy = correct_val / total_val
print(f"\nFinal Training Accuracy: {final_accuracy:.4f}")
print(f"Final Validation Accuracy: {final_val_accuracy:.4f}")
