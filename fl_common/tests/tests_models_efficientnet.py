import os
import torch.nn as nn
from django.test import TestCase
from fl_common.models.efficientnet import get_efficientnet_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingEfficientNetTestCase(TestCase):
    """EfficientNet Model Unit Tests"""
    def test_efficientnet_model(self):
        """
        Test case for validating different configurations of EfficientNet models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
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
        """
        Test case for handling unknown EfficientNet architectures in get_efficientnet_model function.

        Raises:
            AssertionError: If any of the assertions fail.
            ValueError: If an unknown EfficientNet architecture is provided.
        """
        efficientnet_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            get_efficientnet_model(efficientnet_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown EfficientNet Architecture: {efficientnet_type}'
        )
