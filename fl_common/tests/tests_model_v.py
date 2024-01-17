import os
import torch.nn as nn
from django.test import TestCase
from fl_common.models.vgg import get_vgg_model
from fl_common.models.vision_transformer import get_vision_model
from fl_common.models.volo import get_volo_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"

class ProcessingPartVTestCase(TestCase):
    """Volo Model Unit Tests"""
    def test_all_volo_models(self):
        volo_types = ['volo_d1_224', 'volo_d1_384', 'volo_d2_224', 'volo_d2_384', 'volo_d3_224', 'volo_d3_448',
                      'volo_d4_224', 'volo_d4_448', 'volo_d5_224', 'volo_d5_448', 'volo_d5_512']

        for volo_type in volo_types:
            with self.subTest(volo_type=volo_type):
                model = get_volo_model(volo_type, num_classes=10)
                self.assertTrue(isinstance(model, nn.Module))

    def test_unknown_volo_type(self):
        with self.assertRaises(ValueError):
            get_volo_model('unknown_type', num_classes=10)

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

    def test_vgg_unknown_architecture(self):
        vgg_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            get_vgg_model(vgg_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown VGG Architecture : {vgg_type}'
        )

    def test_vgg_last_layer_adaptation(self):
        # Provide a known architecture type
        vgg_type = 'VGG16'
        num_classes = 10

        # Override the last layer with a linear layer for testing purposes
        vgg_model = get_vgg_model(vgg_type, num_classes)
        last_layer = vgg_model.classifier[-1]
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)

    def test_vgg_model_structure(self):
        # Provide a known architecture type
        vgg_type = 'VGG16'
        num_classes = 10

        # Check if the model has a known structure with a linear last layer
        vgg_model = get_vgg_model(vgg_type, num_classes)
        self.assertIsInstance(vgg_model.classifier[-1], nn.Linear)