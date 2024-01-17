import os
import torch.nn as nn
from django.test import TestCase
from fl_common.models.regnet import get_regnet_model
from fl_common.models.resnet import get_resnet_model
from fl_common.models.resnext import get_resnext_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingPartRTestCase(TestCase):
    """ResNext Model Unit Tests"""
    def test_get_resnext_model(self):
        """
        Test case for obtaining various ResNeXt models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # List of ResNeXt model types to test
        resnext_types = ['ResNeXt50_32X4D', 'ResNeXt101_32X8D', 'ResNeXt101_64X4D']
        num_classes = 10  # You can adjust the number of classes as needed

        # Loop through each ResNeXt model type
        for resnext_type in resnext_types:
            with self.subTest(resnext_type=resnext_type):
                # Get the ResNeXt model for testing
                model = get_resnext_model(resnext_type, num_classes)
                # Check if the model is an instance of torch.nn.Module
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_resnext_unknown_architecture(self):
        """
        Test case for handling unknown ResNeXt architecture in get_resnext_model function.

        Raises:
            AssertionError: If the assertion fails.
            ValueError: If an unknown ResNeXt architecture is provided.
        """
        resnext_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            # Attempt to get a ResNeXt model with an unknown architecture
            get_resnext_model(resnext_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown ResNeXt Architecture: {resnext_type}'
        )

    def test_resnext_last_layer_adaptation(self):
        """
        Test case for ensuring last layer adaptation in ResNeXt models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # Provide a known architecture type
        resnext_type = 'ResNeXt50_32X4D'
        num_classes = 10

        # Override the last layer with a linear layer for testing purposes
        resnext_model = get_resnext_model(resnext_type, num_classes)
        last_layer = resnext_model.fc
        # Check if the last layer is an instance of nn.Linear
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)

    """ResNet Mmodel Unit Tests"""
    def test_get_resnet_model(self):
        """
        Test case for obtaining various ResNet models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # List of ResNet model types to test
        resnet_types = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']
        num_classes = 10  # You can adjust the number of classes as needed

        # Loop through each ResNet model type
        for resnet_type in resnet_types:
            with self.subTest(resnet_type=resnet_type):
                # Get the ResNet model for testing
                model = get_resnet_model(resnet_type, num_classes)
                # Check if the model is an instance of torch.nn.Module
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_resnet_unknown_architecture(self):
        """
        Test case for handling unknown ResNet architecture in get_resnet_model function.

        Raises:
            AssertionError: If the assertion fails.
            ValueError: If an unknown ResNet architecture is provided.
        """
        resnet_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            # Attempt to get a ResNet model with an unknown architecture
            get_resnet_model(resnet_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown ResNet Architecture: {resnet_type}'
        )

    def test_resnet_last_layer_adaptation(self):
        """
        Test case for ensuring last layer adaptation in ResNet models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # Provide a known architecture type
        resnet_type = 'ResNet18'
        num_classes = 10

        # Override the last layer with a linear layer for testing purposes
        resnet_model = get_resnet_model(resnet_type, num_classes)
        last_layer = resnet_model.fc
        # Check if the last layer is an instance of nn.Linear
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)

    def test_get_regnet_model(self):
        """
        Test case for obtaining various RegNet models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # List of RegNet model types to test
        regnet_types = ['RegNet_X_400MF', 'RegNet_X_800MF', 'RegNet_X_1_6GF', 'RegNet_X_3_2GF', 'RegNet_X_16GF',
                        'RegNet_Y_400MF', 'RegNet_Y_800MF', 'RegNet_Y_1_6GF', 'RegNet_Y_3_2GF', 'RegNet_Y_16GF']
        num_classes = 10  # You can adjust the number of classes as needed

        # Loop through each RegNet model type
        for regnet_type in regnet_types:
            with self.subTest(regnet_type=regnet_type):
                # Get the RegNet model for testing
                model = get_regnet_model(regnet_type, num_classes)
                # Check if the model is an instance of torch.nn.Module
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_regnet_unknown_architecture(self):
        """
        Test case for handling unknown RegNet architecture in get_regnet_model function.

        Raises:
            AssertionError: If the assertion fails.
            ValueError: If an unknown RegNet architecture is provided.
        """
        regnet_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            # Attempt to get a RegNet model with an unknown architecture
            get_regnet_model(regnet_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown RegNet Architecture: {regnet_type}'
        )

    def test_regnet_last_layer_adaptation(self):
        """
        Test case for ensuring last layer adaptation in RegNet models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # Provide a known architecture type
        regnet_type = 'RegNet_X_400MF'
        num_classes = 10

        # Override the last layer with a linear layer for testing purposes
        regnet_model = get_regnet_model(regnet_type, num_classes)
        last_layer = regnet_model.fc
        # Check if the last layer is an instance of nn.Linear
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)
