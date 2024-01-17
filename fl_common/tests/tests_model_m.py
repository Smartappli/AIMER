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
        # mobilenet = get_mobilenet_model('MobileNet_V3_Small', 1000)
        # self.assertIsNotNone(mobilenet, msg="get_mobilenet_model KO")
        mobilenet_types = ['MobileNet_V2', 'MobileNet_V3_Small', 'MobileNet_V3_Large']
        num_classes = 10  # You can adjust the number of classes as needed

        for mobilenet_type in mobilenet_types:
            with self.subTest(mobilenet_type=mobilenet_type):
                model = get_mobilenet_model(mobilenet_type, num_classes)
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_mobilenet_unknown_architecture(self):
        mobilenet_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            get_mobilenet_model(mobilenet_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown MobileNet Architecture : {mobilenet_type}'
        )

    def test_mobilenet_last_layer_adaptation(self):
        # Provide a known architecture type
        mobilenet_type = 'MobileNet_V2'
        num_classes = 10

        # Override the last layer with a non-linear layer for testing purposes
        mobilenet_model = get_mobilenet_model(mobilenet_type, num_classes)
        last_layer = mobilenet_model.classifier[-1]
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)

    def test_get_mnasnet_model(self):
        # mnasnet = get_mnasnet_model('MNASNet0_5', 1000)
        # self.assertIsNotNone(mnasnet, msg="get_mnasnet_model KO")
        mnasnet_types = ['MNASNet0_5', 'MNASNet0_75', 'MNASNet1_0', 'MNASNet1_3']
        num_classes = 10  # You can adjust the number of classes as needed

        for mnasnet_type in mnasnet_types:
            with self.subTest(mnasnet_type=mnasnet_type):
                model = get_mnasnet_model(mnasnet_type, num_classes)
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_mnasnet_unknown_architecture(self):
        mnasnet_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            get_mnasnet_model(mnasnet_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown MNASNet Architecture: {mnasnet_type}'
        )

    def test_mnasnet_last_layer_adaptation(self):
        # Provide a known architecture type
        mnasnet_type = 'MNASNet0_5'
        num_classes = 10

        # Override the last layer with a non-linear layer for testing purposes
        mnasnet_model = get_mnasnet_model(mnasnet_type, num_classes)
        last_layer = mnasnet_model.classifier[1]
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)

    """MaxVit Model Unit Tests"""
    def test_get_maxvit_model(self):
        # maxvit = get_maxvit_model('MaxVit_T', 1000)
        # self.assertIsNotNone(maxvit, msg="get_maxvit_model KO")
        maxvit_types = ['MaxVit_T']
        num_classes = 10  # You can adjust the number of classes as needed

        for maxvit_type in maxvit_types:
            with self.subTest(maxvit_type=maxvit_type):
                model = get_maxvit_model(maxvit_type, num_classes)
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_maxvit_unknown_architecture(self):
        maxvit_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            get_maxvit_model(maxvit_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown MaxVit Architecture: {maxvit_type}'
        )

    def test_maxvit_last_layer_adaptation(self):
        # Provide a known architecture type
        maxvit_type = 'MaxVit_T'
        num_classes = 10

        # Override the last layer with a non-linear layer for testing purposes
        maxvit_model = get_maxvit_model(maxvit_type, num_classes)
        last_layer = maxvit_model.classifier[-1]
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)