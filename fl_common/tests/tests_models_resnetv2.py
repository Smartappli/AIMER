import os
from django.test import TestCase
from fl_common.models.resnetv2 import get_resnetv2_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingResnetTestCase(TestCase):
    """
    Test case class for processing Resnet models.
    """

    def test_get_resnet_model(self):
        """
        Test case for obtaining various ResNet v2 models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # List of ResNet model types to test
        resnet_types = [
            "resnetv2_50x1_bit",
            "resnetv2_50x3_bit",
            "resnetv2_101x1_bit",
            "resnetv2_101x3_bit",
            "resnetv2_152x2_bit",
            "resnetv2_152x4_bit",
            "resnetv2_50",
            "resnetv2_50d",
            "resnetv2_50t",
            "resnetv2_101",
            "resnetv2_101d",
            "resnetv2_152",
            "resnetv2_152d",
            "resnetv2_50d_gn",
            "resnetv2_50d_evos",
            "resnetv2_50d_frn",
        ]
        num_classes = 10  # You can adjust the number of classes as needed

        # Loop through each ResNet model type
        for resnet_type in resnet_types:
            with self.subTest(resnet_type=resnet_type):
                # Get the ResNet model for testing
                model = get_resnetv2_model(resnet_type, num_classes)
                self.assertIsNotNone(model)
                # Check if the model has the expected number of output classes
                self.assertEqual(model.num_classes, num_classes)

    def test_resnet_unknown_architecture(self):
        """
        Test case for handling unknown ResNet v2 architecture in get_resnet_model function.

        Raises:
            AssertionError: If the assertion fails.
            ValueError: If an unknown ResNet architecture is provided.
        """
        resnet_type = "UnknownArchitecture"
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            # Attempt to get a ResNet model with an unknown architecture
            get_resnetv2_model(resnet_type, num_classes)

        self.assertEqual(str(context.exception),
                         f"Unknown ResNet v2 Architecture: {resnet_type}")
