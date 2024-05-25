import os
from django.test import TestCase
from fl_common.models.vgg import get_vgg_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingVGGTestCase(TestCase):
    """
    Test case class for processing VGG models.
    """

    def test_get_vgg_model(self):
        """
        Test case for obtaining various VGG models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # List of VGG model types to test
        vgg_types = [
            "VGG11",
            "VGG11_BN",
            "VGG13",
            "VGG13_BN",
            "VGG16",
            "VGG16_BN",
            "VGG19",
            "VGG19_BN",
            "vgg11",
            "vgg11_bn",
            "vgg13",
            "vgg13_bn",
            "vgg16",
            "vgg16_bn",
            "vgg19",
            "vgg19_bn",
        ]
        num_classes = 10  # You can adjust the number of classes as needed

        # Loop through each VGG model type
        for vgg_type in vgg_types:
            with self.subTest(vgg_type=vgg_type):
                try:
                    model = get_vgg_model(vgg_type, num_classes)
                    self.assertIsNotNone(model)
                except ValueError:
                    self.fail(f"{vgg_type} should be a known VGG architecture.")

    def test_vgg_unknown_architecture(self):
        """
        Test case for handling unknown VGG architecture in get_vgg_model function.

        Raises:
            AssertionError: If the assertion fails.
            ValueError: If an unknown VGG architecture is provided.
        """
        vgg_type = "UnknownArchitecture"
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            # Attempt to get a VGG model with an unknown architecture
            get_vgg_model(vgg_type, num_classes)

        self.assertEqual(
            str(context.exception), f"Unknown VGG Architecture : {vgg_type}",
        )
