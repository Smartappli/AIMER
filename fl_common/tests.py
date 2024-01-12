import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms, utils
from django.test import TestCase
from fl_common.models.utils import (create_transform,
                                    get_optimizer,
                                    get_criterion,
                                    get_scheduler,
                                    generate_xai_heatmaps,
                                    get_dataset,
                                    EarlyStopping)
from fl_common.models.alexnet import get_alexnet_model
from fl_common.models.convnext import get_convnext_model
from fl_common.models.densenet import get_densenet_model
from fl_common.models.efficientnet import get_efficientnet_model
from fl_common.models.googlenet import get_googlenet_model
from fl_common.models.inception import get_inception_model
from fl_common.models.maxvit import get_maxvit_model
from fl_common.models.mnasnet import get_mnasnet_model
from fl_common.models.mobilenet import get_mobilenet_model
from fl_common.models.regnet import get_regnet_model
from fl_common.models.resnet import get_resnet_model
from fl_common.models.resnext import get_resnext_model
from fl_common.models.shufflenet import get_shufflenet_model
from fl_common.models.squeezenet import get_squeezenet_model
from fl_common.models.swin_transformer import get_swin_model
from fl_common.models.vgg import get_vgg_model
from fl_common.models.vision_transformer import get_vision_model
from fl_common.models.wide_resnet import get_wide_resnet_model


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
            data_augmentation=True
        )

        # self.assertIsInstance(transform, transforms.Compose)
        self.assertTrue(any(isinstance(t, transforms.Resize) for t in transform.transforms))
        self.assertTrue(any(isinstance(t, transforms.CenterCrop) for t in transform.transforms))
        self.assertTrue(any(isinstance(t, transforms.RandomCrop) for t in transform.transforms))
        self.assertTrue(any(isinstance(t, transforms.RandomHorizontalFlip) for t in transform.transforms))
        self.assertTrue(any(isinstance(t, transforms.RandomVerticalFlip) for t in transform.transforms))
        self.assertTrue(any(isinstance(t, transforms.RandomRotation) for t in transform.transforms))
        self.assertTrue(any(isinstance(t, transforms.ColorJitter) for t in transform.transforms))
        self.assertTrue(any(isinstance(t, transforms.GaussianBlur) for t in transform.transforms))
        self.assertTrue(any(isinstance(t, transforms.ToTensor) for t in transform.transforms))
        self.assertTrue(any(isinstance(t, transforms.Normalize) for t in transform.transforms))

    def test_no_to_tensor(self):
        # Test without converting to tensor
        transform = create_transform(resize=(256, 256), to_tensor=False)

        self.assertIsInstance(transform, transforms.Compose)
        self.assertTrue(all(not isinstance(t, transforms.ToTensor) for t in transform.transforms))

    def test_no_normalize(self):
        # Test without normalization
        transform = create_transform(resize=(256, 256), to_tensor=True, normalize=None)

        self.assertIsInstance(transform, transforms.Compose)
        self.assertTrue(all(not isinstance(t, transforms.Normalize) for t in transform.transforms))

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
            'MSELoss', 'L1Loss', 'CTCLoss', 'KLDivLoss', 'GaussianNLLLoss',
            'SmoothL1Loss', 'CrossEntropyLoss', 'BCELoss', 'BCEWithLogitsLoss',
            'NLLLoss', 'PoissonNLLLoss', 'KLDivLoss', 'MarginRankingLoss',
            'HingeEmbeddingLoss', 'MultiLabelMarginLoss', 'SmoothL1Loss',
            'HuberLoss', 'SoftMarginLoss', 'MultiLabelSoftMarginLoss',
            'CosineEmbeddingLoss', 'MultiMarginLoss', 'TripletMarginLoss',
            'TripletMarginWithDistanceLoss'
        ]

        for criterion_name in known_criteria:
            with self.subTest(criterion_name=criterion_name):
                criterion = get_criterion(criterion_name)
                self.assertIsInstance(criterion, nn.Module)

    def test_unknown_criterion(self):
        # Test with an unknown criterion
        with self.assertRaises(ValueError) as context:
            get_criterion('UnknownLoss')

        self.assertEqual(
            str(context.exception),
            'Unknown Criterion : UnknownLoss'
        )

    def test_known_optimizer(self):
        # Test with all known optimizers
        known_optimizers = [
            'SGD', 'Adam', 'RMSprop', 'Adagrad', 'Adadelta', 'AdamW', 'SparseAdam',
            'Adamax', 'ASGD', 'LBFGS', 'Rprop', 'NAdam', 'RAdam'
        ]

        model_parameters = [torch.tensor([1.0], requires_grad=True)]
        learning_rate = 0.001

        for optimizer_name in known_optimizers:
            with self.subTest(optimizer_name=optimizer_name):
                optimizer = get_optimizer(optimizer_name, model_parameters, learning_rate)
                self.assertIsInstance(optimizer, optim.Optimizer)

    def test_unknown_optimizer(self):
        # Test with an unknown optimizer
        model_parameters = [torch.tensor([1.0], requires_grad=True)]
        learning_rate = 0.001

        with self.assertRaises(ValueError) as context:
            get_optimizer('UnknownOptimizer', model_parameters, learning_rate)

        self.assertEqual(
            str(context.exception),
            'Unknown Optimizer : UnknownOptimizer'
        )

    def test_step_scheduler(self):
        # Test with StepLR scheduler
        optimizer = optim.SGD([torch.tensor(1.0, requires_grad=True)], lr=0.001)
        scheduler = get_scheduler(optimizer, scheduler_type='step', step_size=5, gamma=0.1)
        self.assertIsInstance(scheduler, lr_scheduler.StepLR)
        self.assertEqual(scheduler.step_size, 5)
        self.assertEqual(scheduler.gamma, 0.1)

    def test_multi_step_scheduler(self):
        # Test with MultiStepLR scheduler
        optimizer = optim.SGD([torch.tensor(1.0, requires_grad=True)], lr=0.001)
        scheduler = get_scheduler(optimizer, scheduler_type='multi_step', milestones=[5, 10, 15], gamma=0.1)
        self.assertIsInstance(scheduler, lr_scheduler.MultiStepLR)
        # self.assertEqual(scheduler.milestones, [5, 10, 15])
        self.assertEqual(scheduler.gamma, 0.1)

    def test_exponential_scheduler(self):
        # Test with ExponentialLR scheduler
        optimizer = optim.SGD([torch.tensor(1.0, requires_grad=True)], lr=0.001)
        scheduler = get_scheduler(optimizer, scheduler_type='exponential', gamma=0.9)
        self.assertIsInstance(scheduler, lr_scheduler.ExponentialLR)
        self.assertEqual(scheduler.gamma, 0.9)

    def test_invalid_scheduler(self):
        # Test with an invalid scheduler
        optimizer = optim.SGD([torch.tensor(1.0, requires_grad=True)], lr=0.001)
        with self.assertRaises(ValueError) as context:
            get_scheduler(optimizer, scheduler_type='invalid_type', step_size=5, gamma=0.1)

        self.assertEqual(
            str(context.exception),
            'Invalid scheduler_type: invalid_type'
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

    """Wide Resnet Model Unit Tests"""
    def test_get_wide_resnet_model(self):
        wide_resnet_model = get_wide_resnet_model('Wide_ResNet50_2', 1000)
        self.assertIsNotNone(wide_resnet_model, msg="Wide ResNet KO")

    """Vision Transformer Model Unit Tests"""
    def test_get_vision_model(self):
        # vision_model = get_vision_model('ViT_B_16', 1000)
        # self.assertIsNotNone(vision_model, msg="Wision Transform KO")
        vision_types = ['ViT_B_16', 'ViT_B_32', 'ViT_L_16', 'ViT_L_32', 'ViT_H_14']
        num_classes = 10  # You can adjust the number of classes as needed

        for vision_type in vision_types:
            with self.subTest(vision_type=vision_type):
                model = get_vision_model(vision_type, num_classes)
                self.assertIsInstance(model, nn.Module, msg=f'get_maxvit_model {vision_type} KO')

    def test_vision_unknown_architecture(self):
        vision_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            get_vision_model(vision_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown Vision Transformer Architecture: {vision_type}'
        )

    """
    def test_vision_nonlinear_last_layer(self):
        # Provide a vision_type with a known non-linear last layer
        vision_type = 'ViT_B_16'
        num_classes = 10

        # Override the last layer with a non-linear layer for testing purposes
        vision_model = get_vision_model(vision_type, num_classes)
        vision_model.heads[-1] = nn.ReLU()

        with self.assertRaises(ValueError) as context:
            # Try to create the vision model again
            get_vision_model(vision_type, num_classes)

        # Check if the raised ValueError contains the expected message
        self.assertIn(
            'The last layer is not a linear layer.',
            str(context.exception)
        )
    """

    def test_get_vgg_model(self):
        # vgg11 = get_vgg_model('VGG11',1000)
        # self.assertIsNotNone(vgg11, msg="get_vgg_model KO")
        vgg_types = ['VGG11', 'VGG11_BN', 'VGG13', 'VGG13_BN', 'VGG16', 'VGG16_BN', 'VGG19', 'VGG19_BN']
        num_classes = 10  # You can adjust the number of classes as needed

        for vgg_type in vgg_types:
            with self.subTest(vgg_type=vgg_type):
                model = get_vgg_model(vgg_type, num_classes)
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_unknown_architecture(self):
        vgg_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            get_vgg_model(vgg_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown VGG Architecture : {vgg_type}'
        )

    def test_last_layer_adaptation(self):
        # Provide a known architecture type
        vgg_type = 'VGG16'
        num_classes = 10

        # Override the last layer with a linear layer for testing purposes
        vgg_model = get_vgg_model(vgg_type, num_classes)
        last_layer = vgg_model.classifier[-1]
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)

    def test_model_structure(self):
        # Provide a known architecture type
        vgg_type = 'VGG16'
        num_classes = 10

        # Check if the model has a known structure with a linear last layer
        vgg_model = get_vgg_model(vgg_type, num_classes)
        self.assertIsInstance(vgg_model.classifier[-1], nn.Linear)

    def test_get_swim_model(self):
        # swin = get_swin_model('Swin_T', 1000)
        # self.assertIsNotNone(swin, msg="get_swin_model KO")
        swin_types = ['Swin_T', 'Swin_S', 'Swin_B', 'Swin_V2_T', 'Swin_V2_S', 'Swin_V2_B']
        num_classes = 10  # You can adjust the number of classes as needed

        for swin_type in swin_types:
            with self.subTest(swin_type=swin_type):
                model = get_swin_model(swin_type, num_classes)
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_swin_unknown_architecture(self):
        swin_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            get_swin_model(swin_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown DenseNet Architecture : {swin_type}'
        )

    def test_swin_last_layer_adaptation(self):
        # Provide a known architecture type
        swin_type = 'Swin_T'
        num_classes = 10

        # Override the last layer with a linear layer for testing purposes
        swin_model = get_swin_model(swin_type, num_classes)
        last_layer = swin_model.head
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)

    def test_swin_model_structure(self):
        # Provide a known architecture type
        swin_type = 'Swin_T'
        num_classes = 10

        # Check if the model has a known structure with a linear last layer
        swin_model = get_swin_model(swin_type, num_classes)
        self.assertIsInstance(swin_model.head, nn.Linear)

    def test_get_squeezenet_model(self):
        # squeezenet_model = get_squeezenet_model('SqueezeNet1_0', 1000)
        # self.assertIsNotNone(squeezenet_model, msg="get_squeezenet_model KO")
        squeezenet_types = ['SqueezeNet1_0', 'SqueezeNet1_1']
        num_classes = 10  # You can adjust the number of classes as needed

        for squeezenet_type in squeezenet_types:
            with self.subTest(squeezenet_type=squeezenet_type):
                model = get_squeezenet_model(squeezenet_type, num_classes)
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_squeezenet_unknown_architecture(self):
        squeezenet_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            get_squeezenet_model(squeezenet_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown SqueezeNet Architecture: {squeezenet_type}'
        )

    def test_squeezenet_last_layer_adaptation(self):
        # Provide a known architecture type
        squeezenet_type = 'SqueezeNet1_0'
        num_classes = 10

        # Override the last layer with a non-linear layer for testing purposes
        squeezenet_model = get_squeezenet_model(squeezenet_type, num_classes)
        last_layer = squeezenet_model.classifier[1]
        self.assertIsInstance(last_layer, nn.Conv2d)
        self.assertEqual(last_layer.out_channels, num_classes)
        self.assertEqual(last_layer.kernel_size, (1, 1))
        self.assertEqual(last_layer.stride, (1, 1))


    def test_get_shufflenet_model(self):
        # shufflenet = get_shufflenet_model('ShuffleNet_V2_X0_5', 1000)
        # self.assertIsNotNone(shufflenet, msg="get_shufflenet_model KO")
        shufflenet_types = ['ShuffleNet_V2_X0_5', 'ShuffleNet_V2_X1_0', 'ShuffleNet_V2_X1_5', 'ShuffleNet_V2_X2_0']
        num_classes = 10  # You can adjust the number of classes as needed

        for shufflenet_type in shufflenet_types:
            with self.subTest(shufflenet_type=shufflenet_type):
                model = get_shufflenet_model(shufflenet_type, num_classes)
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_shufflenet_unknown_architecture(self):
        shufflenet_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            get_shufflenet_model(shufflenet_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown ShuffleNet Architecture: {shufflenet_type}'
        )

    def test_shufflenet_last_layer_adaptation(self):
        # Provide a known architecture type
        shufflenet_type = 'ShuffleNet_V2_X0_5'
        num_classes = 10

        # Override the last layer with a non-linear layer for testing purposes
        shufflenet_model = get_shufflenet_model(shufflenet_type, num_classes)
        last_layer = shufflenet_model.fc
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)

    """ResNext Model Unit Tests"""
    def test_get_resnext_model(self):
        # resnext_model = get_resnext_model('ResNeXt50_32X4D', 1000)
        # self.assertIsNotNone(resnext_model, msg="get_resnext_model KO")
        resnext_types = ['ResNeXt50_32X4D', 'ResNeXt101_32X8D', 'ResNeXt101_64X4D']
        num_classes = 10  # You can adjust the number of classes as needed

        for resnext_type in resnext_types:
            with self.subTest(resnext_type=resnext_type):
                model = get_resnext_model(resnext_type, num_classes)
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_resnext_unknown_architecture(self):
        resnext_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            get_resnext_model(resnext_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown ResNeXt Architecture: {resnext_type}'
        )

    def test_resnext_last_layer_adaptation(self):
        # Provide a known architecture type
        resnext_type = 'ResNeXt50_32X4D'
        num_classes = 10

        # Override the last layer with a non-linear layer for testing purposes
        resnext_model = get_resnext_model(resnext_type, num_classes)
        last_layer = resnext_model.fc
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)

    """ResNet Mmodel Unit Tests"""
    def test_get_resnet_model(self):
        # resnet = get_resnet_model('ResNet50', 1000)
        # self.assertIsNotNone(resnet, msg="get_resnet_model KO")
        resnet_types = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']
        num_classes = 10  # You can adjust the number of classes as needed

        for resnet_type in resnet_types:
            with self.subTest(resnet_type=resnet_type):
                model = get_resnet_model(resnet_type, num_classes)
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_resnet_unknown_architecture(self):
        resnet_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            get_resnet_model(resnet_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown ResNet Architecture: {resnet_type}'
        )

    def test_resnet_last_layer_adaptation(self):
        # Provide a known architecture type
        resnet_type = 'ResNet18'
        num_classes = 10

        # Override the last layer with a non-linear layer for testing purposes
        resnet_model = get_resnet_model(resnet_type, num_classes)
        last_layer = resnet_model.fc
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)

    def test_get_regnet_model(self):
        # regnet = get_regnet_model('RegNet_X_400MF', 1000)
        # self.assertIsNotNone(regnet,  msg="get_regnet_model KO")
        regnet_types = ['RegNet_X_400MF', 'RegNet_X_800MF', 'RegNet_X_1_6GF', 'RegNet_X_3_2GF', 'RegNet_X_16GF',
                        'RegNet_Y_400MF', 'RegNet_Y_800MF', 'RegNet_Y_1_6GF', 'RegNet_Y_3_2GF', 'RegNet_Y_16GF']
        num_classes = 10  # You can adjust the number of classes as needed

        for regnet_type in regnet_types:
            with self.subTest(regnet_type=regnet_type):
                model = get_regnet_model(regnet_type, num_classes)
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_regnet_unknown_architecture(self):
        regnet_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            get_regnet_model(regnet_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown RegNet Architecture: {regnet_type}'
        )

    def test_regnet_last_layer_adaptation(self):
        # Provide a known architecture type
        regnet_type = 'RegNet_X_400MF'
        num_classes = 10

        # Override the last layer with a non-linear layer for testing purposes
        regnet_model = get_regnet_model(regnet_type, num_classes)
        last_layer = regnet_model.fc
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)

    """MobileNet Model Unit Tests"""
    def test_get_mobilenet_model(self):
        # mobilenet = get_mobilenet_model('MobileNet_V3_Small', 1000)
        # self.assertIsNotNone(mobilenet, msg="get_mobilenet_model KO")
        mobilenet_types = ['MobileNet_V2', 'MobileNet_V3_Small', 'MobileNet_V3_Large']
        num_classes = 10  # You can adjust the number of classes as needed

        for mobilenet_type in mobilenet_types:
            with self.subTest(mobilenet_type=mobilenet_type):
                model = get_mobilenet_model(mobilenet_type, num_classes)
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_mobilenet_unknown_architecture(self):
        mobilenet_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            get_mobilenet_model(mobilenet_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown MobileNet Architecture : {mobilenet_type}'
        )

    def test_mobilenet_last_layer_adaptation(self):
        # Provide a known architecture type
        mobilenet_type = 'MobileNet_V2'
        num_classes = 10

        # Override the last layer with a non-linear layer for testing purposes
        mobilenet_model = get_mobilenet_model(mobilenet_type, num_classes)
        last_layer = mobilenet_model.classifier[-1]
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)

    def test_get_mnasnet_model(self):
        # mnasnet = get_mnasnet_model('MNASNet0_5', 1000)
        # self.assertIsNotNone(mnasnet, msg="get_mnasnet_model KO")
        mnasnet_types = ['MNASNet0_5', 'MNASNet0_75', 'MNASNet1_0', 'MNASNet1_3']
        num_classes = 10  # You can adjust the number of classes as needed

        for mnasnet_type in mnasnet_types:
            with self.subTest(mnasnet_type=mnasnet_type):
                model = get_mnasnet_model(mnasnet_type, num_classes)
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_mnasnet_unknown_architecture(self):
        mnasnet_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            get_mnasnet_model(mnasnet_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown MNASNet Architecture: {mnasnet_type}'
        )

    def test_mnasnet_last_layer_adaptation(self):
        # Provide a known architecture type
        mnasnet_type = 'MNASNet0_5'
        num_classes = 10

        # Override the last layer with a non-linear layer for testing purposes
        mnasnet_model = get_mnasnet_model(mnasnet_type, num_classes)
        last_layer = mnasnet_model.classifier[1]
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)

    """MaxVit Model Unit Tests"""
    def test_get_maxvit_model(self):
        # maxvit = get_maxvit_model('MaxVit_T', 1000)
        # self.assertIsNotNone(maxvit, msg="get_maxvit_model KO")
        maxvit_types = ['MaxVit_T']
        num_classes = 10  # You can adjust the number of classes as needed

        for maxvit_type in maxvit_types:
            with self.subTest(maxvit_type=maxvit_type):
                model = get_maxvit_model(maxvit_type, num_classes)
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_maxvit_unknown_architecture(self):
        maxvit_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            get_maxvit_model(maxvit_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown MaxVit Architecture: {maxvit_type}'
        )

    def test_maxvit_last_layer_adaptation(self):
        # Provide a known architecture type
        maxvit_type = 'MaxVit_T'
        num_classes = 10

        # Override the last layer with a non-linear layer for testing purposes
        maxvit_model = get_maxvit_model(maxvit_type, num_classes)
        last_layer = maxvit_model.classifier[-1]
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)

    """Inception Model Unit Tests"""
    def test_get_inception_model(self):
        # inception = get_inception_model('Inception_V3', 1000)
        # self.assertIsNotNone(inception, msg="get_inception_model KO")
        inception_types = ['Inception_V3']
        num_classes = 10  # You can adjust the number of classes as needed

        for inception_type in inception_types:
            with self.subTest(inception_type=inception_type):
                model = get_inception_model(inception_type, num_classes)
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_inception_unknown_architecture(self):
        inception_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            get_inception_model(inception_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown Inception Architecture: {inception_type}'
        )

    def test_inception_last_layer_adaptation(self):
        # Provide a known architecture type
        inception_type = 'Inception_V3'
        num_classes = 10

        # Override the last layer with a non-linear layer for testing purposes
        inception_model = get_inception_model(inception_type, num_classes)
        last_layer = inception_model.fc
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)

    """GoogleNet Model Unit Tests"""
    def test_get_googlenet_model(self):
        # googlenet = get_googlenet_model('GoogLeNet', 1000)
        # self.assertIsNotNone(googlenet, msg="get_googlenet_model KO")
        googlenet_types = ['GoogLeNet']
        num_classes = 10  # You can adjust the number of classes as needed

        for googlenet_type in googlenet_types:
            with self.subTest(googlenet_type=googlenet_type):
                model = get_googlenet_model(googlenet_type, num_classes)
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_googlenet_unknown_architecture(self):
        googlenet_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            get_googlenet_model(googlenet_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown AlexNet Architecture: {googlenet_type}'
        )

    def test_googlenet_last_layer_adaptation(self):
        # Provide a known architecture type
        googlenet_type = 'GoogLeNet'
        num_classes = 10

        # Override the last layer with a non-linear layer for testing purposes
        googlenet_model = get_googlenet_model(googlenet_type, num_classes)
        last_layer = googlenet_model.fc
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)

    """EfficientNet Model Unit Tests"""
    def test_efficientnet_model(self):
        # efficientnet = get_efficientnet_model('EfficientNetB0', 1000)
        # self.assertIsNotNone(efficientnet, msg="get_efficientnet_model KO")
        efficientnet_types = [
            'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3',
            'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7',
            'EfficientNetV2S', 'EfficientNetV2M', 'EfficientNetV2L'
        ]
        num_classes = 10  # You can adjust the number of classes as needed

        for efficientnet_type in efficientnet_types:
            with self.subTest(efficientnet_type=efficientnet_type):
                model = get_efficientnet_model(efficientnet_type, num_classes)
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_efficientnet_unknown_architecture(self):
        efficientnet_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            get_efficientnet_model(efficientnet_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown DenseNet Architecture : {efficientnet_type}'
        )

    def test_efficientnet_last_layer_adaptation(self):
        # Provide a known architecture type
        efficientnet_type = 'EfficientNetB0'
        num_classes = 10

        # Override the last layer with a non-linear layer for testing purposes
        efficientnet_model = get_efficientnet_model(efficientnet_type, num_classes)
        last_layer = efficientnet_model.classifier[-1]
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)

    '''DenseNet Model Unit Tests'''
    def test_densenet_model(self):
        # densenet = get_densenet_model('DenseNet121', 1000)
        # self.assertIsNotNone(densenet, msg="get_densenet_model KO")
        densenet_types = ['DenseNet121', 'DenseNet161', 'DenseNet169', 'DenseNet201']
        num_classes = 10  # You can adjust the number of classes as needed

        for densenet_type in densenet_types:
            with self.subTest(densenet_type=densenet_type):
                model = get_densenet_model(densenet_type, num_classes)
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_densenet_unknown_architecture(self):
        densenet_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            get_densenet_model(densenet_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown DenseNet Architecture : {densenet_type}'
        )

    def test_denseet_last_layer_adaptation(self):
        # Provide a known architecture type
        densenet_type = 'DenseNet121'
        num_classes = 10

        # Override the last layer with a non-linear layer for testing purposes
        densenet_model = get_densenet_model(densenet_type, num_classes)
        last_layer = densenet_model.classifier
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)

    '''ConvNeXt Model Unit Tests'''
    def test_convnet_model(self):
        # convnext = get_convnext_model('ConvNeXt_Tiny', 1000)
        # self.assertIsNotNone(convnext, msg="get_convnext_model KO")
        convnext_types = ['ConvNeXt_Tiny', 'ConvNeXt_Small', 'ConvNeXt_Base', 'ConvNeXt_Large']
        num_classes = 10  # You can adjust the number of classes as needed

        for convnext_type in convnext_types:
            with self.subTest(convnext_type=convnext_type):
                model = get_convnext_model(convnext_type, num_classes)
                self.assertIsInstance(model, nn.Module)

    def test_convnext_unknown_architecture(self):
        convnext_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            get_convnext_model(convnext_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown DenseNet Architecture : {convnext_type}'
        )

    def test_convnext_last_layer_adaptation(self):
        # Provide a known architecture type
        convnext_type = 'ConvNeXt_Large'
        num_classes = 10

        # Override the last layer with a non-linear layer for testing purposes
        convnext_model = get_convnext_model(convnext_type, num_classes)
        last_layer = None
        for layer in reversed(convnext_model.classifier):
            if isinstance(layer, nn.Linear):
                last_layer = layer
                break

        self.assertIsNotNone(last_layer)
        self.assertEqual(last_layer.out_features, num_classes)

    """AlexNet Model Unit Tests"""
    def test_alexnet_model(self):
        # alexnet = get_alexnet_model('AlexNet', 1000)
        # self.assertIsNotNone(alexnet, msg="get_alexnet_model KO")
        alexnet_types = ['AlexNet']
        num_classes = 10  # You can adjust the number of classes as needed

        for alexnet_type in alexnet_types:
            with self.subTest(alexnet_type=alexnet_type):
                model = get_alexnet_model(alexnet_type, num_classes)
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_alexnet_unknown_architecture(self):
        alexnet_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            get_alexnet_model(alexnet_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown AlexNet Architecture: {alexnet_type}'
        )

    def test_alexnet_last_layer_adaptation(self):
        # Provide a known architecture type
        alexnet_type = 'AlexNet'
        num_classes = 10

        # Override the last layer with a non-linear layer for testing purposes
        alexnet_model = get_alexnet_model(alexnet_type, num_classes)
        last_layer = alexnet_model.classifier[6]
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)
