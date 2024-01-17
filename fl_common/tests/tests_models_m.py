import os
import torch.nn as nn
from django.test import TestCase
from fl_common.models.maxvit import get_maxvit_model
from fl_common.models.mnasnet import get_mnasnet_model
from fl_common.models.mobilenet import get_mobilenet_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingPartMTestCase(TestCase):
    """MobileNet Model Unit Tests"""
    def test_get_mobilenet_model(self):
        """
        Test case for obtaining various MobileNet models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # List of MobileNet model types to test
        mobilenet_types = ['MobileNet_V2', 'MobileNet_V3_Small', 'MobileNet_V3_Large']
        num_classes = 10  # You can adjust the number of classes as needed

        # Loop through each MobileNet model type
        for mobilenet_type in mobilenet_types:
            with self.subTest(mobilenet_type=mobilenet_type):
                # Get the MobileNet model for testing
                model = get_mobilenet_model(mobilenet_type, num_classes)
                # Check if the model is an instance of torch.nn.Module
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_mobilenet_unknown_architecture(self):
        """
        Test case for handling unknown MobileNet architecture in get_mobilenet_model function.

        Raises:
            AssertionError: If the assertion fails.
            ValueError: If an unknown MobileNet architecture is provided.
        """
        mobilenet_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            # Attempt to get a MobileNet model with an unknown architecture
            get_mobilenet_model(mobilenet_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown MobileNet Architecture : {mobilenet_type}'
        )

    def test_mobilenet_last_layer_adaptation(self):
        """
        Test case for ensuring last layer adaptation in MobileNet models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # Provide a known architecture type
        mobilenet_type = 'MobileNet_V2'
        num_classes = 10

        # Override the last layer with a linear layer for testing purposes
        mobilenet_model = get_mobilenet_model(mobilenet_type, num_classes)
        last_layer = mobilenet_model.classifier[-1]
        # Check if the last layer is an instance of nn.Linear
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)

    def test_get_mnasnet_model(self):
        """
        Test case for obtaining various MNASNet models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # List of MNASNet model types to test
        mnasnet_types = ['MNASNet0_5', 'MNASNet0_75', 'MNASNet1_0', 'MNASNet1_3']
        num_classes = 10  # You can adjust the number of classes as needed

        # Loop through each MNASNet model type
        for mnasnet_type in mnasnet_types:
            with self.subTest(mnasnet_type=mnasnet_type):
                # Get the MNASNet model for testing
                model = get_mnasnet_model(mnasnet_type, num_classes)
                # Check if the model is an instance of torch.nn.Module
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_mnasnet_unknown_architecture(self):
        """
        Test case for handling unknown MNASNet architecture in get_mnasnet_model function.

        Raises:
            AssertionError: If the assertion fails.
            ValueError: If an unknown MNASNet architecture is provided.
        """
        mnasnet_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            # Attempt to get an MNASNet model with an unknown architecture
            get_mnasnet_model(mnasnet_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown MNASNet Architecture: {mnasnet_type}'
        )

    def test_mnasnet_last_layer_adaptation(self):
        """
        Test case for ensuring last layer adaptation in MNASNet models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # Provide a known architecture type
        mnasnet_type = 'MNASNet0_5'
        num_classes = 10

        # Override the last layer with a linear layer for testing purposes
        mnasnet_model = get_mnasnet_model(mnasnet_type, num_classes)
        last_layer = mnasnet_model.classifier[1]
        # Check if the last layer is an instance of nn.Linear
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)

    """MaxVit Model Unit Tests"""
    def test_get_maxvit_model(self):
        """
        Test case for obtaining MaxVit models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # List of MaxVit model types to test
        maxvit_types = ['MaxVit_T']
        num_classes = 10  # You can adjust the number of classes as needed

        # Loop through each MaxVit model type
        for maxvit_type in maxvit_types:
            with self.subTest(maxvit_type=maxvit_type):
                # Get the MaxVit model for testing
                model = get_maxvit_model(maxvit_type, num_classes)
                # Check if the model is an instance of torch.nn.Module
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_maxvit_unknown_architecture(self):
        """
        Test case for handling unknown MaxVit architecture in get_maxvit_model function.

        Raises:
            AssertionError: If the assertion fails.
            ValueError: If an unknown MaxVit architecture is provided.
        """
        maxvit_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            # Attempt to get a MaxVit model with an unknown architecture
            get_maxvit_model(maxvit_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown MaxVit Architecture: {maxvit_type}'
        )

    def test_maxvit_last_layer_adaptation(self):
        """
        Test case for ensuring last layer adaptation in MaxVit models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # Provide a known architecture type
        maxvit_type = 'MaxVit_T'
        num_classes = 10

        # Override the last layer with a linear layer for testing purposes
        maxvit_model = get_maxvit_model(maxvit_type, num_classes)
        last_layer = maxvit_model.classifier[-1]
        # Check if the last layer is an instance of nn.Linear
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)
