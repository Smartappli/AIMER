import os
from django.test import TestCase
from fl_common.models.resnet import get_resnet_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingResnetTestCase(TestCase):
    """
    Test case class for processing Resnet models.
    """

    def test_get_resnet_model(self):
        """
        Test case for obtaining various ResNet models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # List of ResNet model types to test
        resnet_types = [
            "ResNet18",
            "ResNet34",
            "ResNet50",
            "ResNet101",
            "ResNet152",
            "resnet10t",
            "resnet14t",
            "resnet18",
            "resnet18d",
            "resnet34",
            "resnet34d",
            "resnet26",
            "resnet26t",
            "resnet26d",
            "resnet50",
            "resnet50c",
            "resnet50d",
            "resnet50s",
            "resnet50t",
            "resnet101",
            "resnet101c",
            "resnet101d",
            "resnet101s",
            "resnet152",
            "resnet152c",
            "resnet152d",
            "resnet152s",
            "resnet200",
            "resnet200d",
            "wide_resnet50_2",
            "wide_resnet101_2",
            "resnet50_gn",
            "resnext50_32x4d",
            "resnext50d_32x4d",
            "resnext101_32x4d",
            "resnext101_32x8d",
            "resnext101_32x16d",
            "resnext101_32x32d",
            "resnext101_64x4d",
            "ecaresnet26t",
            "ecaresnet50d",
            "ecaresnet50d_pruned",
            "ecaresnet50t",
            "ecaresnetlight",
            "ecaresnet101d",
            "ecaresnet101d_pruned",
            "ecaresnet200d",
            "ecaresnet269d",
            "ecaresnext26t_32x4d",
            "ecaresnext50t_32x4d",
            "seresnet18",
            "seresnet34",
            "seresnet50",
            "seresnet50t",
            "seresnet101",
            "seresnet152",
            "seresnet152d",
            "seresnet200d",
            "seresnet269d",
            "seresnext26d_32x4d",
            "seresnext26t_32x4d",
            "seresnext50_32x4d",
            "seresnext101_32x8d",
            "seresnext101d_32x8d",
            "seresnext101_64x4d",
            "senet154",
            "resnetblur18",
            "resnetblur50",
            "resnetblur50d",
            "resnetblur101d",
            "resnetaa34d",
            "resnetaa50",
            "resnetaa50d",
            "resnetaa101d",
            "seresnetaa50d",
            "seresnextaa101d_32x8d",
            "seresnextaa201d_32x8d",
            "resnetrs50",
            "resnetrs101",
            "resnetrs152",
            "resnetrs200",
            "resnetrs270",
            "resnetrs350",
            "resnetrs420",
        ]
        num_classes = 10  # You can adjust the number of classes as needed

        # Loop through each ResNet model type
        for resnet_type in resnet_types:
            with self.subTest(resnet_type=resnet_type):
                # Get the ResNet model for testing
                model = get_resnet_model(resnet_type, num_classes)
                try:
                    model = get_resnet_model(resnet_type, num_classes)
                    self.assertIsNotNone(model)
                except ValueError:
                    self.fail(
                        f"{resnet_type} should be a known Resnet architecture.")

    def test_resnet_unknown_architecture(self):
        """
        Test case for handling unknown ResNet architecture in get_resnet_model function.

        Raises:
            AssertionError: If the assertion fails.
            ValueError: If an unknown ResNet architecture is provided.
        """
        resnet_type = "UnknownArchitecture"
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            # Attempt to get a ResNet model with an unknown architecture
            get_resnet_model(resnet_type, num_classes)

        self.assertEqual(str(context.exception),
                         f"Unknown ResNet Architecture: {resnet_type}")
