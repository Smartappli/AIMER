import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# from tqdm import tqdm
from captum.attr import (
    Saliency,
    IntegratedGradients,
    GuidedBackprop,
    DeepLift,
    # LayerConductance,
    # NeuronConductance,
    Occlusion,
    ShapleyValueSampling,
)
# from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# import seaborn as sns
# import time


def create_transform(resize=None,
                     center_crop=None,
                     random_crop=None,
                     random_horizontal_flip=False,
                     random_vertical_flip=False,
                     to_tensor=True,
                     normalize=None,
                     random_rotation=None,
                     color_jitter=None,
                     gaussian_blur=None,
                     data_augmentation=False,
                     rotation_range=45,
                     horizontal_flip_prob=0.5,
                     vertical_flip_prob=0.5):
    """
    Create a PyTorch transform based on the specified parameters.

    Parameters:
    - resize: Tuple or int, size of the resized image (width, height) or single size for both dimensions.
    - center_crop: Tuple or int, size of the center crop (width, height) or single size for both dimensions.
    - random_crop: Tuple or int, size of the random crop (width, height) or single size for both dimensions.
    - random_horizontal_flip: bool, whether to apply random horizontal flip.
    - random_vertical_flip: bool, whether to apply random vertical flip.
    - to_tensor: bool, whether to convert the image to a PyTorch tensor.
    - normalize: Tuple, mean and standard deviation for normalization.
    - random_rotation: float, range of degrees for random rotation.
    - color_jitter: Tuple, parameters for color jittering (brightness, contrast, saturation, hue).
    - gaussian_blur: Tuple, parameters for Gaussian blur (kernel size, sigma).
    - data_augmentation: bool, whether to apply additional data augmentation.
    - rotation_range: float, range of degrees for random rotation during data augmentation.
    - horizontal_flip_prob: float, probability of random horizontal flip during data augmentation.
    - vertical_flip_prob: float, probability of random vertical flip during data augmentation.

    Returns:
    - transform: torchvision.transforms.Compose, a composition of specified transformations.
    """
    transform_list = []

    if data_augmentation:
        if resize is not None:
            if isinstance(resize, int):
                resize = (resize, resize)
            transform_list.append(transforms.Resize(resize))

        if center_crop is not None:
            if isinstance(center_crop, int):
                center_crop = (center_crop, center_crop)
            transform_list.append(transforms.CenterCrop(center_crop))

        if random_crop is not None:
            if isinstance(random_crop, int):
                random_crop = (random_crop, random_crop)
            transform_list.append(transforms.RandomCrop(random_crop))

        if random_horizontal_flip:
            transform_list.append(transforms.RandomHorizontalFlip(p=horizontal_flip_prob))

        if random_vertical_flip:
            transform_list.append(transforms.RandomVerticalFlip(p=vertical_flip_prob))

        if random_rotation is not None:
            transform_list.append(transforms.RandomRotation(degrees=rotation_range))

        if color_jitter is not None:
            transform_list.append(transforms.ColorJitter(*color_jitter))

        if gaussian_blur is not None:
            transform_list.append(transforms.GaussianBlur(gaussian_blur))

        if to_tensor:
            transform_list.append(transforms.ToTensor())

        if normalize is not None:
            transform_list.append(transforms.Normalize(mean=normalize[0], std=normalize[1]))

    transform = transforms.Compose(transform_list)
    return transform


def get_dataset(dataset_path, batch_size, augmentation_params, normalize_params):
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

    num_classes = len(dataset.classes)

    return train_loader, val_loader, test_loader, num_classes

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

    # List of available XAI methods in captum
    xai_methods = [
        Saliency(model),
        IntegratedGradients(model),
        GuidedBackprop(model),
        DeepLift(model),
        # LayerConductance(model.features[7], model),
        # NeuronConductance(model.features[7], model),
        Occlusion(model),
        ShapleyValueSampling(model),
    ]

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

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=7, verbose=False):
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