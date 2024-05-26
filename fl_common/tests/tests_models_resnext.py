import os

from django.test import TestCase
from torch import nn

from fl_common.models.resnext import get_resnext_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingResnextTestCase(TestCase):
    """
    Test case class for processing Resnext models.
    """

    def test_get_resnext_model(self):
        """
        Test case for obtaining various ResNeXt models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # List of ResNeXt model types to test
        resnext_types = [
            "ResNeXt50_32X4D",
            "ResNeXt101_32X8D",
            "ResNeXt101_64X4D",
        ]
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
        resnext_type = "UnknownArchitecture"
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            # Attempt to get a ResNeXt model with an unknown architecture
            get_resnext_model(resnext_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f"Unknown ResNeXt Architecture: {resnext_type}",
        )

    def test_resnext_last_layer_adaptation(self):
        """
        Test case for ensuring last layer adaptation in ResNeXt models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # Provide a known architecture type
        resnext_type = "ResNeXt50_32X4D"
        num_classes = 10

        # Override the last layer with a linear layer for testing purposes
        resnext_model = get_resnext_model(resnext_type, num_classes)
        last_layer = resnext_model.fc
        # Check if the last layer is an instance of nn.Linear
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)
