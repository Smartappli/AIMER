import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
from captum.attr import (
    Saliency,
    IntegratedGradients,
    GuidedBackprop,
    DeepLift,
    LayerConductance,
    NeuronConductance,
    Occlusion,
    ShapleyValueSampling,
)
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Parameters
dataset_path = 'c:/IA/Data'  # Replace with the actual path to your dataset
batch_size = 32

vgg_type = 'VGG16_BN'
best_val_loss = float('inf')  # Initialize the best validation loss

perform_second_training = True  # Set to True to perform the second training
perform_third_training = False  # Set To True to perform the third training
verbose = True

optimizer_name_phase1 = 'SGD'
learning_rate_phase1 = 0.01
criterion_name_phase1 = 'CrossEntropyLoss'
num_epochs_phase1 = 3  # Number of epochs for the first training phase
scheduler_phase1 = False
early_stopping_patience_phase1 = 5

optimizer_name_phase2 = 'SGD'
learning_rate_phase2 = 0.0005
criterion_name_phase2 = 'CrossEntropyLoss'
num_epochs_phase2 = 50  # Number of epochs for the second training phase
scheduler_phase2 = False
early_stopping_patience_phase2 = 5

optimizer_name_phase3 = 'SGD'
learning_rate_phase3 = 0.0001
criterion_name_phase3 = 'CrossEntropyLoss'
num_epochs_phase3 = 50  # Number of epochs for the third training phase
scheduler_phase3 = False
early_stopping_patience_phase3 = 5


def get_vgg_model(vgg_type='VGG16', num_classes=1000):
    # Load the pre-trained version of VGG
    if vgg_type == 'VGG11':
        weights = models.VGG11_Weights.DEFAULT
        vgg_model = models.vgg11(weights=weights)
    elif vgg_type == 'VGG11_BN':
        weights = models.VGG11_BN_Weights.DEFAULT
        vgg_model = models.vgg11_bn(weights=weights)
    elif vgg_type == 'VGG13':
        weights = models.VGG13_Weights.DEFAULT
        vgg_model = models.vgg13(weights=weights)
    elif vgg_type == 'VGG13_BN':
        weights = models.VGG13_BN_Weights.DEFAULT
        vgg_model = models.vgg13_bn(weights=weights)
    elif vgg_type == 'VGG16':
        weights = models.VGG16_Weights.DEFAULT
        vgg_model = models.vgg16(weights=weights)
    elif vgg_type == 'VGG16_BN':
        weights = models.VGG16_BN_Weights.DEFAULT
        vgg_model = models.vgg16_bn(weights=weights)
    elif vgg_type == 'VGG19':
        weights = models.VGG19_Weights.DEFAULT
        vgg_model = models.vgg19(weights=weights)
    elif vgg_type == 'VGG19_BN':
        weights = models.VGG19_BN_Weights.DEFAULT
        vgg_model = models.vgg19_bn(weights=weights)
    else:
        raise ValueError(f'Unknown VGG Architecture : {vgg_type}')

    # Modify last layer to suit number of classes
    num_features = vgg_model.classifier[-1].in_features
    vgg_model.classifier[-1] = nn.Linear(num_features, num_classes)

    return vgg_model


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


# Function to generate and save XAI heatmap for a specific image using selected methods
def generate_xai_heatmaps(model, image_tensor, label, save_dir, methods=None):
    model.eval()

    # Create input tensor with batch dimension
    input_tensor = image_tensor.unsqueeze(0)

    # Use the specified methods or all available methods if None
    methods = methods or xai_methods

    for method in methods:
        # Compute attributions
        attributions = method.attribute(input_tensor, target=label)

        # Take the absolute value of the attributions
        attributions = torch.abs(attributions)

        # Reduce attributions to 2D (assuming the input is an image)
        attributions = attributions.sum(dim=1)

        # Normalize attributions to [0, 1]
        attributions = (attributions - attributions.min()) / (attributions.max() - attributions.min())

        # Convert to numpy array for plotting
        attributions_np = attributions.squeeze(0).cpu().detach().numpy()

        # Plot and save the heatmap
        method_name = str(method).split('.')[-1].split(' ')[0]
        plt.imshow(attributions_np, cmap='viridis')
        plt.title(f'XAI Heatmap for {method_name} (Label: {label})')
        plt.colorbar()
        save_path = os.path.join(save_dir, f'xai_heatmap_{method_name}_{label}.png')
        plt.savefig(save_path)
        plt.show()


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
model = get_vgg_model(vgg_type=vgg_type, num_classes=num_classes)

# List of available XAI methods in captum
xai_methods = [
    Saliency(model),
    IntegratedGradients(model),
    GuidedBackprop(model),
    DeepLift(model),
    LayerConductance(model.features[7], model),
    NeuronConductance(model.features[7], model),
    Occlusion(model),
    ShapleyValueSampling(model),
]

# Move the model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model = model.to(device)

# Define the loss criterion and optimizer
model_parameters = model.parameters()
criterion = get_criterion(criterion_name_phase1)
optimizer = get_optimizer(optimizer_name_phase1, model_parameters, learning_rate_phase1)

# scheduler = get_scheduler(optimizer, scheduler_type='step', step_size=10, gamma=0.5)
# scheduler = get_scheduler(optimizer, scheduler_type='multi_step', milestones=[30, 60, 90], gamma=0.5)
# scheduler = get_scheduler(optimizer, scheduler_type='exponential', gamma=0.95)

scheduler = get_scheduler(optimizer, scheduler_type='step', step_size=10, gamma=0.5)

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=5, verbose=verbose):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


# Training loop
early_stopping_phase1 = EarlyStopping(patience=early_stopping_patience_phase1)
early_stopping_phase2 = EarlyStopping(patience=early_stopping_patience_phase2)
early_stopping_phase3 = EarlyStopping(patience=early_stopping_patience_phase3)

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
elapsed_times = []

start_time = time.time()

for epoch in range(num_epochs_phase1):
    epoch_start_time = time.time()

    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    print(f"\nEpoch {epoch + 1}/{num_epochs_phase1}, Learning Rate: {optimizer.param_groups[0]['lr']}")

    # Use tqdm for a progress bar over the training batches
    with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs_phase1}") as progress_bar:
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
    if scheduler_phase1:
        scheduler.step()

    # Early stopping check
    early_stopping_phase1(avg_val_loss, model)
    if early_stopping_phase1.early_stop:
        print("Early stopping after {} epochs".format(epoch + 1))
        break

    # Save the model if the current validation loss is the best
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.pth')

    epoch_end_time = time.time()
    elapsed_time = epoch_end_time - epoch_start_time
    elapsed_times.append(elapsed_time)

    # Print and plot results
    print(f"Epoch {epoch + 1}/{num_epochs_phase1} => "
          f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
          f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, "
          f"Elapsed Time: {elapsed_time:.2f} seconds")

if perform_second_training and not early_stopping_phase1.early_stop:  # Proceed only if the first phase didn't early stop
    print("\nStarting the second training phase...\n")

    # Optionally reset the optimizer, criterion and scheduler for the second phase
    model_parameters = model.parameters()
    criterion = get_criterion(criterion_name_phase2)
    optimizer = get_optimizer(optimizer_name_phase2, model_parameters, learning_rate_phase2)

    # scheduler = get_scheduler(optimizer, scheduler_type='step', step_size=10, gamma=0.5)
    # scheduler = get_scheduler(optimizer, scheduler_type='multi_step', milestones=[30, 60, 90], gamma=0.5)
    # scheduler = get_scheduler(optimizer, scheduler_type='exponential', gamma=0.95)

    scheduler = get_scheduler(optimizer, scheduler_type='step', step_size=10, gamma=0.5)

    for epoch in range(num_epochs_phase2):
        epoch_start_time = time.time()

        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        print(f"\nEpoch {epoch + 1}/{num_epochs_phase2}, Learning Rate: {optimizer.param_groups[0]['lr']}")

        # Use tqdm for a progress bar over the training batches
        with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs_phase2}") as progress_bar:
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
        if scheduler_phase2:
            scheduler.step()

        # Early stopping check for the second phase
        early_stopping_phase2(avg_val_loss, model)
        if early_stopping_phase2.early_stop:
            print("Early stopping after the second training phase.")
            break

        # Save the model if the current validation loss is the best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        epoch_end_time = time.time()
        elapsed_time = epoch_end_time - epoch_start_time
        elapsed_times.append(elapsed_time)

        # Print and plot results for the second phase
        print(f"\nEpoch {epoch + 1}/{num_epochs_phase2} (Phase 2) => "
              f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, "
              f"Elapsed Time: {elapsed_time:.2f} seconds")

# Third Training Phase (Optional)
perform_third_training = True  # Set to True to perform the third training

if perform_third_training and not early_stopping_phase2.early_stop:  # Proceed only if the second phase didn't early stop
    print("\nStarting the third training phase...\n")

    # Optionally reset the optimizer, criterion and scheduler for the second phase
    model_parameters = model.parameters()
    criterion = get_criterion(criterion_name_phase3)
    optimizer = get_optimizer(optimizer_name_phase3, model_parameters, learning_rate_phase3)

    # scheduler = get_scheduler(optimizer, scheduler_type='step', step_size=10, gamma=0.5)
    # scheduler = get_scheduler(optimizer, scheduler_type='multi_step', milestones=[30, 60, 90], gamma=0.5)
    # scheduler = get_scheduler(optimizer, scheduler_type='exponential', gamma=0.95)

    scheduler = get_scheduler(optimizer, scheduler_type='step', step_size=10, gamma=0.5)

    for epoch in range(num_epochs_phase3):
        epoch_start_time = time.time()

        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        print(f"\nEpoch {epoch + 1}/{num_epochs_phase3}, Learning Rate: {optimizer.param_groups[0]['lr']}")

        # Use tqdm for a progress bar over the training batches
        with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs_phase3}") as progress_bar:
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
        if scheduler_phase3:
            scheduler.step()

        # Early stopping check for the third phase
        early_stopping_phase3(avg_val_loss, model)
        if early_stopping_phase3.early_stop:
            print("Early stopping after the third training phase.")
            break

        # Save the model if the current validation loss is the best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        epoch_end_time = time.time()
        elapsed_time = epoch_end_time - epoch_start_time
        elapsed_times.append(elapsed_time)

        # Print and plot results for the third phase
        print(f"\nEpoch {epoch + 1}/{num_epochs_phase3} (Phase 3) => "
              f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, "
              f"Elapsed Time: {elapsed_time:.2f} seconds")

# Calculate total training time
total_training_time = time.time() - start_time
print(f"\nTotal Training Time: {total_training_time / 60:.2f} minutes")

# Plot training curves
plt.figure(figsize=(12, 6))

# Plot losses
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot accuracies
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracies')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png')  # Saving the graph

plt.tight_layout()
plt.show()

# Test the final model on the test set
model.eval()
correct_test = 0
total_test = 0

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Print final results
test_accuracy = correct_test / total_test
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Generate and plot confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap=plt.cm.Blues, cbar=False, annot_kws={"size": 14})
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix.png')  # Saving the confusion matrix
plt.show()  # Confusion matrix display

# Print classification report
class_names = dataset.classes
class_report = classification_report(all_labels, all_preds, target_names=class_names)
print("\nClassification Report:\n", class_report)

# Save classification report to a text file
with open('classification_report.txt', 'w') as report_file:
    report_file.write("Classification Report:\n" + class_report)

# Loop through test dataset and generate XAI heatmaps for specific methods
for i, (inputs, labels) in enumerate(test_loader):
    inputs, labels = inputs.to(device), labels.to(device)

    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)

    if predicted != labels:
        save_dir = 'xai_heatmaps'
        os.makedirs(save_dir, exist_ok=True)

        # Specify the methods you want to use (e.g., 'GuidedBackprop' and 'IntegratedGradients')
        specific_methods = [GuidedBackprop(model), IntegratedGradients(model)]

        generate_xai_heatmaps(model, inputs[0], labels.item(), save_dir=save_dir, methods=specific_methods)