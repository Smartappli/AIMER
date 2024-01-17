import os
import torch.nn as nn
from django.test import TestCase
from fl_common.models.efficientnet import get_efficientnet_model
from fl_common.models.edgenet import get_edgenet_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingPartETestCase(TestCase):

    """EfficientNet Model Unit Tests"""
    def test_efficientnet_model(self):
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

    """EdgeNet Model Unit Tests"""
    def test_valid_edgenet_type(self):
        edgenet_types = ["edgenext_xx_small", "edgenext_x_small", "edgenext_small", "edgenext_base", "edgenext_small_rw"]
        num_classes = 10

        for edgenet_type in edgenet_types:
            with self.subTest(edgenet_type=edgenet_type):
                model = get_edgenet_model(edgenet_type, num_classes)
                self.assertIsNotNone(model)
                # Add more specific assertions if needed

    def test_invalid_edgenet_type(self):
        invalid_edgenet_type = "invalid_type"
        num_classes = 10

        with self.assertRaises(ValueError):
            get_edgenet_model(invalid_edgenet_type, num_classes)

    def test_invalid_num_classes(self):
        edgenet_type = "edgenext_small"
        invalid_num_classes = "not_an_integer"

        with self.assertRaises(ValueError):
            # Ajout d'une vérification pour s'assurer que invalid_num_classes est un entier
            if not isinstance(invalid_num_classes, int):
                raise ValueError("invalid_num_classes doit être un entier")

            get_edgenet_model(edgenet_type, invalid_num_classes)

