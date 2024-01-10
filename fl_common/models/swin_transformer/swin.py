import os
import torch
import torch.nn as nn
# import torch.optim as optim
# import torch.optim.lr_scheduler as lr_scheduler
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
from fl_common.models.utils import get_optimizer, get_criterion, get_scheduler, generate_xai_heatmaps, create_transform

# Dataset Parameters
dataset_path = 'c:/IA/Data'  # Replace with the actual path to the dataset
normalize_params = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
augmentation_params = {
    'data_augmentation': True,
    'random_rotation': 0.8,
    'rotation_range': 90,  # Augmentation plus importante de la rotation
    'horizontal_flip_prob': 0.8,  # Probabilité plus élevée de retournement horizontal
    'vertical_flip_prob': 0.8,  # Probabilité plus élevée de retournement vertical
    'resize': 224,
}
batch_size = 32

# Model Parameters
swin_type = 'Swin_V2_B'
best_val_loss = float('inf')  # Initialize the best validation loss
save_dir = 'c:/TFE/Models/' + swin_type + '/'  # Replace with the actual path where to save results
os.makedirs(save_dir, exist_ok=True)

# Training Parameters
perform_second_training = True  # Set to True to perform the second training
perform_third_training = False  # Set To True to perform the third training
verbose = True

optimizer_name_phase1 = 'SGD'
learning_rate_phase1 = 0.01
criterion_name_phase1 = 'CrossEntropyLoss'
num_epochs_phase1 = 5  # Number of epochs for the first training phase
scheduler_phase1 = False
early_stopping_patience_phase1 = 5

optimizer_name_phase2 = 'SGD'
learning_rate_phase2 = 0.0005
criterion_name_phase2 = 'CrossEntropyLoss'
num_epochs_phase2 = 50  # Number of epochs for the second training phase
scheduler_phase2 = True
early_stopping_patience_phase2 = 5

optimizer_name_phase3 = 'SGD'
learning_rate_phase3 = 0.0001
criterion_name_phase3 = 'CrossEntropyLoss'
num_epochs_phase3 = 50  # Number of epochs for the third training phase
scheduler_phase3 = False
early_stopping_patience_phase3 = 5

xai = True

def get_swin_model(swin_type='Swin_T', num_classes=1000):
    # Load the pre-trained version of DenseNet
    if swin_type == 'Swin_T':
        weights = models.Swin_T_Weights.DEFAULT
        swin_model = models.swin_t(weights=weights)
    elif swin_type == 'Swin_S':
        weights = models.Swin_S_Weights.DEFAULT
        swin_model = models.swin_s(weights=weights)
    elif swin_type == 'Swin_B':
        weights = models.Swin_B_Weights.DEFAULT
        swin_model = models.swin_b(weights=weights)
    elif swin_type == 'Swin_V2_T':
        weights = models.Swin_V2_T_Weights.DEFAULT
        swin_model = models.swin_v2_t(weights=weights)
    elif swin_type == 'Swin_V2_S':
        weights = models.Swin_V2_S_Weights.DEFAULT
        swin_model = models.swin_v2_s(weights=weights)
    elif swin_type == 'Swin_V2_B':
        weights = models.Swin_V2_B_Weights.DEFAULT
        swin_model = models.swin_v2_b(weights=weights)
    else:
        raise ValueError(f'Unknown DenseNet Architecture : {swin_type}')

    # Modify last layer to suit number of classes
    if hasattr(swin_model, 'head') and isinstance(swin_model.head, nn.Linear):
        num_features = swin_model.head.in_features
        swin_model.head = nn.Linear(num_features, num_classes)
    else:
        raise ValueError('Model does not have a known structure.')

    return swin_model

# Load your custom dataset
dataset = datasets.ImageFolder(root=dataset_path,
                               transform=create_transform(**augmentation_params,
                                                          normalize=normalize_params))

# Split the dataset into training and testing sets
train_indices, test_indices = train_test_split(list(range(len(dataset))), test_size=0.1, random_state=42)

# Further split the training set into training and validation sets
train_indices, val_indices = train_test_split(train_indices, test_size=0.222222, random_state=42)

# Create SubsetRandomSampler for each set
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

# Calculate the number of images in each set
total_images = len(dataset)
num_train_images = len(train_indices)
num_val_images = len(val_indices)
num_test_images = len(test_indices)

# Print the information
print(f"Nombre total d'images dans le dataset: {total_images}")
print(f"Nombre d'images dans l'ensemble d'entraînement: {num_train_images}")
print(f"Nombre d'images dans l'ensemble de validation: {num_val_images}")
print(f"Nombre d'images dans l'ensemble de test: {num_test_images}")

# Define data loaders
train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

# Use the pre-trained VGG model
num_classes = len(dataset.classes)
model = get_swin_model(swin_type=swin_type, num_classes=num_classes)

# List of available XAI methods in captum
xai_methods = [
    Saliency(model),
    IntegratedGradients(model),
    GuidedBackprop(model),
    DeepLift(model),
    LayerConductance(model.head, model),
    NeuronConductance(model.head, model),
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
    with tqdm(enumerate(train_loader), total=len(train_loader),
              desc=f"Epoch {epoch + 1}/{num_epochs_phase1}") as progress_bar:
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
        torch.save(model.state_dict(), save_dir + 'best_model.pth')

    epoch_end_time = time.time()
    elapsed_time = epoch_end_time - epoch_start_time
    elapsed_times.append(elapsed_time)

    # Print and plot results
    elapsed_time_msg = "Elapsed Time: "

    if elapsed_time >= 3600:
        elapsed_time_msg += f"{elapsed_time / 3600:.2f} hour(s)"
    elif elapsed_time >= 60:
        elapsed_time_msg += f"{elapsed_time / 60:.2f} minute(s)"
    else:
        elapsed_time_msg += f"{elapsed_time:.2f} seconds"

    print(f"Epoch {epoch + 1}/{num_epochs_phase1} => "
          f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
          f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, "
          f"{elapsed_time_msg}")

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
        with tqdm(enumerate(train_loader), total=len(train_loader),
                  desc=f"Epoch {epoch + 1}/{num_epochs_phase2}") as progress_bar:
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
            torch.save(model.state_dict(), save_dir + 'best_model.pth')

        epoch_end_time = time.time()
        elapsed_time = epoch_end_time - epoch_start_time
        elapsed_times.append(elapsed_time)

        # Print and plot results for the second phase
        elapsed_time_msg = "Elapsed Time: "

        if elapsed_time >= 3600:
            elapsed_time_msg += f"{elapsed_time / 3600:.2f} hour(s)"
        elif elapsed_time >= 60:
            elapsed_time_msg += f"{elapsed_time / 60:.2f} minute(s)"
        else:
            elapsed_time_msg += f"{elapsed_time:.2f} seconds"

        print(f"\nEpoch {epoch + 1}/{num_epochs_phase2} (Phase 2) => "
              f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, "
              f"{elapsed_time_msg}")

# Third Training Phase (Optional)
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
        with tqdm(enumerate(train_loader), total=len(train_loader),
                  desc=f"Epoch {epoch + 1}/{num_epochs_phase3}") as progress_bar:
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
            torch.save(model.state_dict(), save_dir + 'best_model.pth')

        epoch_end_time = time.time()
        elapsed_time = epoch_end_time - epoch_start_time
        elapsed_times.append(elapsed_time)

        # Print and plot results for the third phase
        elapsed_time_msg = "Elapsed Time: "

        if elapsed_time >= 3600:
            elapsed_time_msg += f"{elapsed_time / 3600:.2f} hour(s)"
        elif elapsed_time >= 60:
            elapsed_time_msg += f"{elapsed_time / 60:.2f} minute(s)"
        else:
            elapsed_time_msg += f"{elapsed_time:.2f} seconds"

        print(f"\nEpoch {epoch + 1}/{num_epochs_phase3} (Phase 3) => "
              f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, "
              f"{elapsed_time_msg}")

# Calculate total training time
total_training_time = time.time() - start_time
if total_training_time >= 3600:
    unit, time_value = 'hour(s)', total_training_time / 3600
elif total_training_time >= 60:
    unit, time_value = 'minute(s)', total_training_time / 60
else:
    unit, time_value = 'seconds', total_training_time

print(f"\nTotal Training Time: {time_value:.2f} {unit}")

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot losses
axes[0].plot(train_losses, label='Training Loss')
axes[0].plot(val_losses, label='Validation Loss')
axes[0].set_title('Training and Validation Losses')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

# Plot accuracies
axes[1].plot(train_accuracies, label='Training Accuracy')
axes[1].plot(val_accuracies, label='Validation Accuracy')
axes[1].set_title('Training and Validation Accuracies')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

fig.tight_layout()

# Saving the graph
save_path = os.path.join(save_dir, 'training_curves.png')
fig.savefig(save_path)

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
plt.savefig(save_dir + 'confusion_matrix.png')  # Saving the confusion matrix
plt.show()  # Confusion matrix display

# Print classification report
class_names = dataset.classes
class_report = classification_report(all_labels, all_preds, target_names=class_names)
print("\nClassification Report:\n", class_report)

# Save classification report to a text file
with open(save_dir + 'classification_report.txt', 'w') as report_file:
    report_file.write(save_dir + "Classification Report:\n" + class_report)

# Loop through test dataset and generate XAI heatmaps for specific methods
if xai:
    save_dir += 'xai_heatmaps/'
    # Loop through test dataset and generate XAI heatmaps for specific methods
    for i, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        # Convert predicted and labels to scalar values
        predicted_scalars = predicted.tolist()  # Convert to list
        labels_scalars = labels.tolist()  # Convert to list

        for j, (predicted_scalar, label_scalar) in enumerate(zip(predicted_scalars, labels_scalars)):
            if predicted_scalar != label_scalar:
                print(
                    f"Example {i * test_loader.batch_size + j + 1}: Prediction: {predicted_scalar}, Actual: {label_scalar}")

                # Specify the methods you want to use (e.g., 'GuidedBackprop' and 'IntegratedGradients')
                specific_methods = [GuidedBackprop(model), IntegratedGradients(model)]

                # Create a directory for XAI heatmaps based on the specific example
                example_dir = f"{save_dir}/example_{i * test_loader.batch_size + j + 1}/"
                os.makedirs(example_dir, exist_ok=True)

                generate_xai_heatmaps(model, inputs[j], label_scalar, save_dir=example_dir, methods=specific_methods)
