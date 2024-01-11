import os
import time
import torch
import torch.nn as nn
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
import matplotlib.pyplot as plt
import seaborn as sns
from fl_common.models.utils import (get_optimizer,
                                    get_criterion,
                                    get_scheduler,
                                    generate_xai_heatmaps,
                                    get_dataset,
                                    EarlyStopping)
from fl_common.models.convnext.convnext import get_convnext_model
from fl_common.models.densenet.densenet import get_densenet_model
from fl_common.models.efficientnet.efficientnet import get_efficientnet_model
from fl_common.models.mobilenet.mobilenet import get_mobilenet_model
from fl_common.models.regnet.regnet import get_regnet_model
from fl_common.models.resnet.resnet import get_resnet_model
from fl_common.models.swin_transformer.swin import get_swin_model
from fl_common.models.vgg.vgg import get_vgg_model

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

model_list = ['resnet18', 'Swin_V2_T', 'RegNet_X_400MF', 'MobileNet_V3_Small',
              'ConvNeXt_Tiny', 'VGG11', 'DenseNet121', 'EfficientNetB0']
# Model Parameters
best_val_loss = float('inf')  # Initialize the best validation loss

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


def get_familly_model(model_type, num_classes):
    """
    Returns a model based on the specified model family and type.

    Parameters:
    - model_type (str): Type of the model within the specified family.

    Returns:
    - model (torch.nn.Module): Model corresponding to the specified family and type.

    Raises:
    - ValueError: If the specified model family is not recognized.
    """

    if model_type in ['VGG11', 'VGG11_BN', 'VGG13', 'VGG13_BN', 'VGG16', 'VGG16_BN', 'VGG19', 'VGG19_BN']:
        model = get_vgg_model(model_type, num_classes)
    elif model_type in ['Swin_T', 'Swin_S', 'Swin_B', 'Swin_V2_T', 'Swin_V2_S', 'Swin_V2_B']:
        model = get_swin_model(model_type, num_classes)
    elif model_type in ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 'ResNeXt50_32X4D',
                        'ResNeXt101_32X4D', 'ResNeXt101_64X4D', 'Wide_ResNet50_2', 'Wide_ResNet101_2']:
        model = get_resnet_model(model_type, num_classes)
    elif model_type in ['RegNet_X_400MF', 'RegNet_X_800MF', 'RegNet_X_1_6GF', 'RegNet_X_3_2GF', 'RegNet_X_16GF',
                        'RegNet_Y_400MF', 'RegNet_Y_800MF', 'RegNet_Y_1_6GF', 'RegNet_Y_3_2GF', 'RegNet_Y_16GF']:
        model = get_regnet_model(model_type, num_classes)
    elif model_type in ['MobileNet_V2', 'MobileNet_V3_Small', 'MobileNet_V3_Large']:
        model = get_mobilenet_model(model_type, num_classes)
    elif model_type in ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4',
                        'EfficientNetB5', 'EfficientNetB0', 'EfficientNetB7', 'EfficientNetV2S', 'EfficientNetV',
                        'EfficientNetV2L']:
        model = get_efficientnet_model(model_type, num_classes)
    elif model_type in ['DenseNet121', 'DenseNet161', 'DenseNet169', 'DenseNet201']:
        model = get_densenet_model(model_type, num_classes)
    elif model_type in ['ConvNeXt_Tiny', 'ConvNeXt_Small', 'ConvNeXt_Base', 'ConvNeXt_Large']:
        model = get_convnext_model(model_type, num_classes)
    else:
        model = get_vgg_model(vgg_type='VGG16', num_classes=4)
    return model


for model_type in model_list:
    save_dir = 'c:/TFE/Models/' + model_type + '/'  # Replace with the actual path where to save results
    os.makedirs(save_dir, exist_ok=True)

    print(f"Model: {model_type}")

    # Load your custom dataset
    train_loader, val_loader, test_loader, num_classes, class_names = get_dataset(dataset_path,
                                                                                  batch_size,
                                                                                  augmentation_params,
                                                                                  normalize_params)

    # Use the pre-trained model
    model = get_familly_model(model_type, num_classes)

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

    # Training loop
    early_stopping_phase1 = EarlyStopping(patience=early_stopping_patience_phase1, verbose=verbose)
    early_stopping_phase2 = EarlyStopping(patience=early_stopping_patience_phase2, verbose=verbose)
    early_stopping_phase3 = EarlyStopping(patience=early_stopping_patience_phase3, verbose=verbose)

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
                        progress_bar.set_postfix(Batch_Loss=f"{avg_batch_loss:.4f}",
                                                 Batch_Accuracy=f"{batch_accuracy:.4f}")

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
                        progress_bar.set_postfix(Batch_Loss=f"{avg_batch_loss:.4f}",
                                                 Batch_Accuracy=f"{batch_accuracy:.4f}")

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

                    generate_xai_heatmaps(model, inputs[j], label_scalar, save_dir=example_dir,
                                          methods=specific_methods)
