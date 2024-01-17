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
        # resnext_model = get_resnext_model('ResNeXt50_32X4D', 1000)
        # self.assertIsNotNone(resnext_model, msg="get_resnext_model KO")
        resnext_types = ['ResNeXt50_32X4D', 'ResNeXt101_32X8D', 'ResNeXt101_64X4D']
        num_classes = 10  # You can adjust the number of classes as needed

        for resnext_type in resnext_types:
            with self.subTest(resnext_type=resnext_type):
                model = get_resnext_model(resnext_type, num_classes)
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_resnext_unknown_architecture(self):
        resnext_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            get_resnext_model(resnext_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown ResNeXt Architecture: {resnext_type}'
        )

    def test_resnext_last_layer_adaptation(self):
        # Provide a known architecture type
        resnext_type = 'ResNeXt50_32X4D'
        num_classes = 10

        # Override the last layer with a non-linear layer for testing purposes
        resnext_model = get_resnext_model(resnext_type, num_classes)
        last_layer = resnext_model.fc
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)

    """ResNet Mmodel Unit Tests"""
    def test_get_resnet_model(self):
        # resnet = get_resnet_model('ResNet50', 1000)
        # self.assertIsNotNone(resnet, msg="get_resnet_model KO")
        resnet_types = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']
        num_classes = 10  # You can adjust the number of classes as needed

        for resnet_type in resnet_types:
            with self.subTest(resnet_type=resnet_type):
                model = get_resnet_model(resnet_type, num_classes)
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_resnet_unknown_architecture(self):
        resnet_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            get_resnet_model(resnet_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown ResNet Architecture: {resnet_type}'
        )

    def test_resnet_last_layer_adaptation(self):
        # Provide a known architecture type
        resnet_type = 'ResNet18'
        num_classes = 10

        # Override the last layer with a non-linear layer for testing purposes
        resnet_model = get_resnet_model(resnet_type, num_classes)
        last_layer = resnet_model.fc
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)

    def test_get_regnet_model(self):
        # regnet = get_regnet_model('RegNet_X_400MF', 1000)
        # self.assertIsNotNone(regnet,  msg="get_regnet_model KO")
        regnet_types = ['RegNet_X_400MF', 'RegNet_X_800MF', 'RegNet_X_1_6GF', 'RegNet_X_3_2GF', 'RegNet_X_16GF',
                        'RegNet_Y_400MF', 'RegNet_Y_800MF', 'RegNet_Y_1_6GF', 'RegNet_Y_3_2GF', 'RegNet_Y_16GF']
        num_classes = 10  # You can adjust the number of classes as needed

        for regnet_type in regnet_types:
            with self.subTest(regnet_type=regnet_type):
                model = get_regnet_model(regnet_type, num_classes)
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_regnet_unknown_architecture(self):
        regnet_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            get_regnet_model(regnet_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown RegNet Architecture: {regnet_type}'
        )

    def test_regnet_last_layer_adaptation(self):
        # Provide a known architecture type
        regnet_type = 'RegNet_X_400MF'
        num_classes = 10

        # Override the last layer with a non-linear layer for testing purposes
        regnet_model = get_regnet_model(regnet_type, num_classes)
        last_layer = regnet_model.fc
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)