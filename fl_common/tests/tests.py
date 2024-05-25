import os
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torchvision import transforms, utils
from django.test import TestCase
from fl_common.models.utils import (
    create_transform,
    get_optimizer,
    get_criterion,
    get_scheduler,
)

# generate_xai_heatmaps,
# get_dataset,
# EarlyStopping

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingTestCase(TestCase):
    """
    def test_basic_transform(self):
        # Test with basic transformation settings
        transform = create_transform(resize=(256, 256), to_tensor=True, normalize=(0.5, 0.5))

        self.assertIsInstance(transform, transforms.Compose)
        self.assertTrue(any(isinstance(t, transforms.Resize) for t in transform.transforms))
        self.assertTrue(any(isinstance(t, transforms.ToTensor) for t in transform.transforms))
        self.assertTrue(any(isinstance(t, transforms.Normalize) for t in transform.transforms))
    """

    def test_data_augmentation_transform(self):
        # Test with data augmentation settings
        transform = create_transform(
            resize=(256, 256),
            center_crop=(224, 224),
            random_crop=(200, 200),
            random_horizontal_flip=True,
            random_vertical_flip=True,
            to_tensor=True,
            normalize=(0.5, 0.5),
            random_rotation=30,
            color_jitter=(0.2, 0.2, 0.2, 0.2),
            gaussian_blur=(5, 1),
            data_augmentation=True,
        )

        # self.assertIsInstance(transform, transforms.Compose)
        self.assertTrue(
            any(
                isinstance(t, transforms.Resize)
                for t in transform.transforms
            ),
        )
        self.assertTrue(
            any(
                isinstance(t, transforms.CenterCrop)
                for t in transform.transforms
            ),
        )
        self.assertTrue(
            any(
                isinstance(t, transforms.RandomCrop)
                for t in transform.transforms
            ),
        )
        self.assertTrue(
            any(
                isinstance(t, transforms.RandomHorizontalFlip)
                for t in transform.transforms
            ),
        )
        self.assertTrue(
            any(
                isinstance(t, transforms.RandomVerticalFlip)
                for t in transform.transforms
            ),
        )
        self.assertTrue(
            any(
                isinstance(t, transforms.RandomRotation)
                for t in transform.transforms
            ),
        )
        self.assertTrue(
            any(
                isinstance(t, transforms.ColorJitter)
                for t in transform.transforms
            ),
        )
        self.assertTrue(
            any(
                isinstance(t, transforms.GaussianBlur)
                for t in transform.transforms
            ),
        )
        self.assertTrue(
            any(
                isinstance(t, transforms.ToTensor)
                for t in transform.transforms
            ),
        )
        self.assertTrue(
            any(
                isinstance(t, transforms.Normalize)
                for t in transform.transforms
            ),
        )

    def test_no_to_tensor(self):
        # Test without converting to tensor
        transform = create_transform(resize=(256, 256), to_tensor=False)

        self.assertIsInstance(transform, transforms.Compose)
        self.assertTrue(
            all(
                not isinstance(t, transforms.ToTensor)
                for t in transform.transforms
            ),
        )

    def test_no_normalize(self):
        # Test without normalization
        transform = create_transform(
            resize=(256, 256),
            to_tensor=True,
            normalize=None,
        )

        self.assertIsInstance(transform, transforms.Compose)
        self.assertTrue(
            all(
                not isinstance(t, transforms.Normalize)
                for t in transform.transforms
            ),
        )

    """
    def test_get_dataset(self):
        # Mock dataset directory and parameters
        dataset_path = '/test/dataset'
        batch_size = 32
        augmentation_params = {'resize': (256, 256), 'random_crop': (224, 224)}
        normalize_params = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

        # Call the function
        train_loader, val_loader, test_loader, num_classes, class_names = get_dataset(
            dataset_path, batch_size, augmentation_params, normalize_params)

        # Check if DataLoader objects are returned
        self.assertIsInstance(train_loader, utils.data.DataLoader)
        self.assertIsInstance(val_loader, utils.data.DataLoader)
        self.assertIsInstance(test_loader, utils.data.DataLoader)

        # Check if the number of classes and class names are integers and lists, respectively
        self.assertIsInstance(num_classes, int)
        self.assertIsInstance(class_names, list)

        # Check if the number of images in each set is a positive integer
        self.assertGreater(len(train_loader), 0)
        self.assertGreater(len(val_loader), 0)
        self.assertGreater(len(test_loader), 0)

    def test_transform_creation(self):
        # Mock dataset directory and parameters
        dataset_path = '/path/to/dataset'
        batch_size = 32
        augmentation_params = {'resize': (256, 256), 'random_crop': (224, 224)}
        normalize_params = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

        # Call the create_transform function
        transform = create_transform(**augmentation_params, normalize=normalize_params)

        # Check if the transform is a torchvision.transforms.Compose object
        self.assertIsInstance(transform, transforms.Compose)
    """

    def test_known_criterion(self):
        # Test with all known criteria
        known_criteria = [
            "MSELoss",
            "L1Loss",
            "CTCLoss",
            "KLDivLoss",
            "GaussianNLLLoss",
            "SmoothL1Loss",
            "CrossEntropyLoss",
            "BCELoss",
            "BCEWithLogitsLoss",
            "NLLLoss",
            "PoissonNLLLoss",
            "KLDivLoss",
            "MarginRankingLoss",
            "HingeEmbeddingLoss",
            "MultiLabelMarginLoss",
            "SmoothL1Loss",
            "HuberLoss",
            "SoftMarginLoss",
            "MultiLabelSoftMarginLoss",
            "CosineEmbeddingLoss",
            "MultiMarginLoss",
            "TripletMarginLoss",
            "TripletMarginWithDistanceLoss",
        ]

        for criterion_name in known_criteria:
            with self.subTest(criterion_name=criterion_name):
                criterion = get_criterion(criterion_name)
                self.assertIsInstance(criterion, nn.Module)

    def test_unknown_criterion(self):
        # Test with an unknown criterion
        with self.assertRaises(ValueError) as context:
            get_criterion("UnknownLoss")

        self.assertEqual(
            str(context.exception),
            "Unknown Criterion: UnknownLoss",
        )

    def test_known_optimizer(self):
        # Test with all known optimizers
        known_optimizers = [
            "SGD",
            "Adam",
            "RMSprop",
            "Adagrad",
            "Adadelta",
            "AdamW",
            "SparseAdam",
            "Adamax",
            "ASGD",
            "LBFGS",
            "Rprop",
            "NAdam",
            "RAdam",
        ]

        model_parameters = [torch.tensor([1.0], requires_grad=True)]
        learning_rate = 0.001

        for optimizer_name in known_optimizers:
            with self.subTest(optimizer_name=optimizer_name):
                optimizer = get_optimizer(
                    optimizer_name,
                    model_parameters,
                    learning_rate,
                )
                self.assertIsInstance(optimizer, optim.Optimizer)

    def test_unknown_optimizer(self):
        # Test with an unknown optimizer
        model_parameters = [torch.tensor([1.0], requires_grad=True)]
        learning_rate = 0.001

        with self.assertRaises(ValueError) as context:
            get_optimizer("UnknownOptimizer", model_parameters, learning_rate)

        self.assertEqual(
            str(context.exception),
            "Unknown Optimizer: UnknownOptimizer",
        )

    def test_step_scheduler(self):
        # Test with StepLR scheduler
        optimizer = optim.SGD([torch.tensor(1.0, requires_grad=True)], lr=0.001)
        scheduler = get_scheduler(
            optimizer,
            scheduler_type="step",
            step_size=5,
            gamma=0.1,
        )
        self.assertIsInstance(scheduler, lr_scheduler.StepLR)
        self.assertEqual(scheduler.step_size, 5)
        self.assertEqual(scheduler.gamma, 0.1)

    def test_multi_step_scheduler(self):
        # Test with MultiStepLR scheduler
        optimizer = optim.SGD([torch.tensor(1.0, requires_grad=True)], lr=0.001)
        scheduler = get_scheduler(
            optimizer,
            scheduler_type="multi_step",
            milestones=[5, 10, 15],
            gamma=0.1,
        )
        self.assertIsInstance(scheduler, lr_scheduler.MultiStepLR)
        # self.assertEqual(scheduler.milestones, [5, 10, 15])
        self.assertEqual(scheduler.gamma, 0.1)

    def test_exponential_scheduler(self):
        # Test with ExponentialLR scheduler
        optimizer = optim.SGD([torch.tensor(1.0, requires_grad=True)], lr=0.001)
        scheduler = get_scheduler(
            optimizer,
            scheduler_type="exponential",
            gamma=0.9,
        )
        self.assertIsInstance(scheduler, lr_scheduler.ExponentialLR)
        self.assertEqual(scheduler.gamma, 0.9)

    def test_invalid_scheduler(self):
        # Test with an invalid scheduler
        optimizer = optim.SGD([torch.tensor(1.0, requires_grad=True)], lr=0.001)
        with self.assertRaises(ValueError) as context:
            get_scheduler(
                optimizer, scheduler_type="invalid_type", step_size=5, gamma=0.1,
            )

        self.assertEqual(
            str(context.exception),
            "Invalid scheduler_type: invalid_type",
        )

    """
    def test_early_stopping(self):
        # Test early stopping behavior

        # Initialize EarlyStopping object
        early_stopping = EarlyStopping(patience=3, verbose=True)

        # Simulate training loop
        val_losses = [0.9, 0.8, 0.7, 0.7, 0.8, 0.9, 1.0]

        # Check early stopping criteria after each epoch
        for epoch, val_loss in enumerate(val_losses):
            with self.subTest(epoch=epoch):
                early_stopping(val_loss, model=None)

                # If the best loss is not None, it should be decreasing
                if early_stopping.best_loss is not None:
                    self.assertLessEqual(val_loss, early_stopping.best_loss)

                # If the validation loss increases, the counter should increase
                if val_loss > early_stopping.best_loss:
                    self.assertEqual(early_stopping.counter, epoch + 1)

                # If the validation loss decreases, the counter should reset
                if val_loss < early_stopping.best_loss:
                    self.assertEqual(early_stopping.counter, 0)

                # If the counter exceeds patience, early_stop should be triggered
                if early_stopping.counter >= early_stopping.patience:
                    self.assertTrue(early_stopping.early_stop)
                    break
    """
