import os
from torch import nn
from django.test import TestCase
from fl_common.models.mnasnet import get_mnasnet_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingMNasnetTestCase(TestCase):
    """
    Test case class for processing Mnasnet models.
    """

    def test_get_mnasnet_model(self):
        """
        Test case for obtaining various MNASNet models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # List of MNASNet model types to test
        mnasnet_types = ["MNASNet0_5", "MNASNet0_75", "MNASNet1_0", "MNASNet1_3"]
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
        mnasnet_type = "UnknownArchitecture"
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            # Attempt to get an MNASNet model with an unknown architecture
            get_mnasnet_model(mnasnet_type, num_classes)

        self.assertEqual(
            str(context.exception), f"Unknown MNASNet Architecture: {mnasnet_type}"
        )

    def test_mnasnet_last_layer_adaptation(self):
        """
        Test case for ensuring last layer adaptation in MNASNet models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # Provide a known architecture type
        mnasnet_type = "MNASNet0_5"
        num_classes = 10

        # Override the last layer with a linear layer for testing purposes
        mnasnet_model = get_mnasnet_model(mnasnet_type, num_classes)
        last_layer = mnasnet_model.classifier[1]
        # Check if the last layer is an instance of nn.Linear
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)
