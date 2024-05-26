import os
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
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


def maybe_add_transform(transform_list, condition, transform, *args, **kwargs):
    """Helper function to add a transform to the list if the condition is met."""
    if condition:
        # Convert single integers to tuples for resize, crop operations
        if isinstance(condition, int) and transform in {
            transforms.Resize,
            transforms.CenterCrop,
            transforms.RandomCrop,
        }:
            condition = (condition, condition)
        transform_list.append(transform(*args, **kwargs))


def create_transform(
    resize=None,
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
    vertical_flip_prob=0.5,
):
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
        maybe_add_transform(
            transform_list, resize, transforms.Resize, size=resize
        )
        maybe_add_transform(
            transform_list, center_crop, transforms.CenterCrop, size=center_crop
        )
        maybe_add_transform(
            transform_list, random_crop, transforms.RandomCrop, size=random_crop
        )
        maybe_add_transform(
            transform_list,
            random_horizontal_flip,
            transforms.RandomHorizontalFlip,
            p=horizontal_flip_prob,
        )
        maybe_add_transform(
            transform_list,
            random_vertical_flip,
            transforms.RandomVerticalFlip,
            p=vertical_flip_prob,
        )
        maybe_add_transform(
            transform_list,
            random_rotation,
            transforms.RandomRotation,
            degrees=rotation_range,
        )
        maybe_add_transform(
            transform_list, color_jitter, transforms.ColorJitter, *color_jitter
        )
        maybe_add_transform(
            transform_list,
            gaussian_blur,
            transforms.GaussianBlur,
            kernel_size=gaussian_blur,
        )
        maybe_add_transform(transform_list, to_tensor, transforms.ToTensor)
        maybe_add_transform(
            transform_list,
            normalize,
            transforms.Normalize,
            mean=normalize[0],
            std=normalize[1],
        )

    return transforms.Compose(transform_list)


def get_dataset(
    dataset_path,
    batch_size,
    augmentation_params,
    normalize_params,
):
    """
    Create and configure data loaders for a custom dataset.

    Args:
    - dataset_path (str): The path to the root directory of the custom dataset.
    - batch_size (int): The batch size for training, validation, and testing data loaders.
    - augmentation_params (dict): A dictionary containing parameters for data augmentation.
    - normalize_params (dict): A dictionary containing parameters for data normalization.

    Returns:
    - train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
    - val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
    - test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
    - num_classes (int): The number of classes in the dataset.
    - class_names (list): List of class names present in the dataset.

    Note:
    - The dataset is loaded using torchvision.datasets.ImageFolder.
    - It is split into training, validation, and test sets using train_test_split.
    - SubsetRandomSampler is used to create samplers for each set.
    - The number of images in each set is printed.
    - Data loaders are created for each set with the specified batch size.
    - The number of classes and class names are determined and returned.

    Example Usage:
    ```python
    dataset_path = '/path/to/dataset'
    batch_size = 32
    augmentation_params = {'resize': (256, 256), 'random_crop': (224, 224)}
    normalize_params = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

    train_loader, val_loader, test_loader, num_classes, class_names = get_dataset(
        dataset_path, batch_size, augmentation_params, normalize_params)
    ```
    """

    # Load your custom dataset
    dataset = datasets.ImageFolder(
        root=dataset_path,
        transform=create_transform(
            **augmentation_params,
            normalize=normalize_params,
        ),
    )

    # Split the dataset into training and testing sets
    train_indices, test_indices = train_test_split(
        list(range(len(dataset))),
        test_size=0.1,
        random_state=42,
    )

    # Further split the training set into training and validation sets
    train_indices, val_indices = train_test_split(
        train_indices,
        test_size=0.222222,
        random_state=42,
    )

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
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
    )
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=test_sampler,
    )

    num_classes = len(dataset.classes)
    class_names = dataset.classes

    return train_loader, val_loader, test_loader, num_classes, class_names


def get_criterion(criterion_name):
    """
    Return the specified loss criterion from PyTorch's nn module.

    Args:
    - criterion_name (str): Name of the loss criterion to be instantiated.

    Returns:
    - criterion (torch.nn.modules.loss._Loss): The instantiated loss criterion.

    Raises:
    - ValueError: If the provided criterion_name is not recognized.

    Supported Loss Criteria:
    - 'MSELoss': Mean Squared Error Loss
    - 'L1Loss': L1 Loss (Mean Absolute Error)
    - 'CTCLoss': Connectionist Temporal Classification Loss
    - 'KLDivLoss': Kullback-Leibler Divergence Loss
    - 'GaussianNLLLoss': Gaussian Negative Log Likelihood Loss
    - 'SmoothL1Loss': Smooth L1 Loss (Huber Loss)
    - 'CrossEntropyLoss': Cross-Entropy Loss
    - 'BCELoss': Binary Cross-Entropy Loss
    - 'BCEWithLogitsLoss': Binary Cross-Entropy with Logits Loss
    - 'NLLLoss': Negative Log Likelihood Loss
    - 'PoissonNLLLoss': Poisson Negative Log Likelihood Loss
    - 'MarginRankingLoss': Margin Ranking Loss
    - 'HingeEmbeddingLoss': Hinge Embedding Loss
    - 'MultiLabelMarginLoss': Multi-Label Margin Loss
    - 'HuberLoss': Huber Loss (Smooth L1 Loss)
    - 'SoftMarginLoss': Soft Margin Loss
    - 'MultiLabelSoftMarginLoss': Multi-Label Soft Margin Loss
    - 'CosineEmbeddingLoss': Cosine Embedding Loss
    - 'MultiMarginLoss': Multi-Margin Loss
    - 'TripletMarginLoss': Triplet Margin Loss
    - 'TripletMarginWithDistanceLoss': Triplet Margin with Distance Loss

    Example Usage:
    ```python
    criterion_name = 'CrossEntropyLoss'
    criterion = get_criterion(criterion_name)
    ```
    """
    criterion_dict = {
        "MSELoss": nn.MSELoss,
        "L1Loss": nn.L1Loss,
        "CTCLoss": nn.CTCLoss,
        "KLDivLoss": nn.KLDivLoss,
        "GaussianNLLLoss": nn.GaussianNLLLoss,
        "SmoothL1Loss": nn.SmoothL1Loss,
        "CrossEntropyLoss": nn.CrossEntropyLoss,
        "BCELoss": nn.BCELoss,
        "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
        "NLLLoss": nn.NLLLoss,
        "PoissonNLLLoss": nn.PoissonNLLLoss,
        "MarginRankingLoss": nn.MarginRankingLoss,
        "HingeEmbeddingLoss": nn.HingeEmbeddingLoss,
        "MultiLabelMarginLoss": nn.MultiLabelMarginLoss,
        "HuberLoss": nn.HuberLoss,
        "SoftMarginLoss": nn.SoftMarginLoss,
        "MultiLabelSoftMarginLoss": nn.MultiLabelSoftMarginLoss,
        "CosineEmbeddingLoss": nn.CosineEmbeddingLoss,
        "MultiMarginLoss": nn.MultiMarginLoss,
        "TripletMarginLoss": nn.TripletMarginLoss,
        "TripletMarginWithDistanceLoss": nn.TripletMarginWithDistanceLoss,
    }

    try:
        return criterion_dict[criterion_name]()
    except KeyError as exc:
        msg = f"Unknown Criterion: {criterion_name}"
        raise ValueError(msg) from exc


def get_optimizer(optimizer_name, model_parameters, learning_rate):
    """
    Return the specified optimizer from PyTorch's optim module.

    Args:
    - optimizer_name (str): Name of the optimizer to be instantiated.
    - model_parameters (iterable): Iterable of model parameters to optimize.
    - learning_rate (float): Learning rate for the optimizer.

    Returns:
    - optimizer (torch.optim.Optimizer): The instantiated optimizer.

    Raises:
    - ValueError: If the provided optimizer_name is not recognized.

    Supported Optimizers:
    - 'SGD': Stochastic Gradient Descent
    - 'Adam': Adam Optimizer
    - 'RMSprop': RMSprop Optimizer
    - 'Adagrad': Adagrad Optimizer
    - 'Adadelta': Adadelta Optimizer
    - 'AdamW': Adam with Weight Decay Optimizer
    - 'SparseAdam': Sparse Adam Optimizer
    - 'Adamax': Adamax Optimizer
    - 'ASGD': Accelerated Stochastic Gradient Descent
    - 'LBFGS': Limited-memory Broyden-Fletcher-Goldfarb-Shanno
    - 'Rprop': Resilient Backpropagation
    - 'NAdam': Nesterov Adam Optimizer
    - 'RAdam': Rectified Adam Optimizer

    Example Usage:
    ```python
    optimizer_name = 'Adam'
    model_parameters = model.parameters()
    learning_rate = 0.001
    optimizer = get_optimizer(optimizer_name, model_parameters, learning_rate)
    ```
    """
    optimizer_dict = {
        "SGD": optim.SGD,
        "Adam": optim.Adam,
        "RMSprop": optim.RMSprop,
        "Adagrad": optim.Adagrad,
        "Adadelta": optim.Adadelta,
        "AdamW": optim.AdamW,
        "SparseAdam": optim.SparseAdam,
        "Adamax": optim.Adamax,
        "ASGD": optim.ASGD,
        "LBFGS": optim.LBFGS,
        "Rprop": optim.Rprop,
        "NAdam": optim.NAdam,
        "RAdam": optim.RAdam,
    }

    try:
        return optimizer_dict[optimizer_name](
            model_parameters,
            lr=learning_rate,
        )
    except KeyError as exc:
        msg = f"Unknown Optimizer: {optimizer_name}"
        raise ValueError(msg) from exc


def get_scheduler(optimizer, scheduler_type="step", **kwargs):
    """
    Get a learning rate scheduler for the optimizer.

    Parameters:
        optimizer (torch.optim.Optimizer): Optimizer for which the scheduler is desired.
        scheduler_type (str): Type of scheduler to use. Options: 'step', 'multi_step', 'exponential'.
        **kwargs: Additional arguments for the specific scheduler type.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: Learning rate scheduler.

    Raises:
        ValueError: If the provided scheduler_type is not recognized.
    """
    scheduler_dict = {
        "step": lr_scheduler.StepLR,
        "multi_step": lr_scheduler.MultiStepLR,
        "exponential": lr_scheduler.ExponentialLR,
    }

    try:
        return scheduler_dict[scheduler_type](optimizer, **kwargs)
    except KeyError as exc:
        msg = f"Invalid scheduler_type: {scheduler_type}"
        raise ValueError(msg) from exc


def generate_xai_heatmaps(model, image_tensor, label, save_dir, methods=None):
    """
    Generate and save XAI (Explainable AI) heatmaps using various attribution methods.

    Args:
    - model (torch.nn.Module): The trained PyTorch model for which heatmaps are generated.
    - image_tensor (torch.Tensor): Input image tensor for which explanations are generated.
    - label (int): The target label for which attributions are computed.
    - save_dir (str): Directory to save the generated heatmaps.
    - methods (list, optional): List of XAI methods from captum library. Default is None,
      which uses a predefined set of available methods.

    Returns:
    - None

    Note:
    - This function uses captum library for computing attributions and generating heatmaps.
    - The input image tensor should have the same dimensions as the model input.
    - The generated heatmaps are saved in the specified directory with filenames indicating
      the XAI method and the target label.

    Example Usage:
    ```python
    model = MyModel()
    image_tensor = load_image('path/to/image.jpg')
    label = 3
    save_dir = 'path/to/save/heatmaps'

    # Generate heatmaps using default methods
    generate_xai_heatmaps(model, image_tensor, label, save_dir)

    # Generate heatmaps using specific methods
    custom_methods = [Saliency(model), GuidedBackprop(model)]
    generate_xai_heatmaps(model, image_tensor, label, save_dir, methods=custom_methods)
    ```
    """
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
        attributions = (attributions - attributions.min()) / (
            attributions.max() - attributions.min()
        )

        # Convert to numpy array for plotting
        attributions_np = attributions.squeeze(0).cpu().detach().numpy()

        # Plot and save the heatmap
        # method_name = str(method).split('.')[-1].split(' ')[0]
        method_name = str(method).rsplit(".", maxsplit=1)[-1].split(" ")[0]
        plt.imshow(attributions_np, cmap="viridis")
        plt.title(f"XAI Heatmap for {method_name} (Label: {label})")
        plt.colorbar()
        save_path = save_dir + "/" + f"xai_heatmap_{method_name}_{label}.png"
        plt.savefig(save_path)
        plt.show()


# Early stopping class
class EarlyStopping:
    """
    Monitor validation loss during training and stop the training process early
    if the validation loss does not improve for a specified number of consecutive epochs.

    Args:
    - patience (int): Number of epochs with no improvement after which training will be stopped.
    - verbose (bool): If True, prints a message each time the validation loss plateaus.

    Attributes:
    - patience (int): Number of epochs with no improvement allowed.
    - verbose (bool): If True, prints a message each time the validation loss plateaus.
    - counter (int): Counter to keep track of consecutive epochs without improvement.
    - best_loss (float): The best validation loss observed during training.
    - early_stop (bool): Flag to indicate whether early stopping criteria are met.

    Methods:
    - __call__(val_loss, model): Updates the early stopping criteria based on the provided validation loss.
      Should be called after each epoch during training.

    Example Usage:
    ```python
    # Initialize EarlyStopping object
    early_stopping = EarlyStopping(patience=5, verbose=True)

    # Inside the training loop
    for epoch in range(num_epochs):
        # Training steps here

        # Validate the model and get the validation loss
        val_loss = validate(model, val_loader)

        # Check early stopping criteria
        early_stopping(val_loss, model)

        # Break the loop if early stopping criteria are met
        if early_stopping.early_stop:
            print("Early stopping triggered. Training halted.")
            break
    ```
    """

    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        """
        Update early stopping criteria based on the provided validation loss.

        Args:
        - val_loss (float): Validation loss obtained during the current epoch.
        - model (torch.nn.Module): The trained PyTorch model being monitored.

        Returns:
        - None
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.verbose:
                print(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}",
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
