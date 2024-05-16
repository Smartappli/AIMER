import os
from django.test import TestCase
from fl_common.models.rexnet import get_rexnet_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingRexnetTestCase(TestCase):
    """
    Test case class for processing RexNet models.
    """

    def test_known_architectures(self):
        """
        Test if the function returns a valid Rexnet model for known Rexnet architectures.
        """
        architectures = [
            "rexnet_100",
            "rexnet_130",
            "rexnet_150",
            "rexnet_200",
            "rexnet_300",
            "rexnetr_100",
            "rexnetr_130",
            "rexnetr_150",
            "rexnetr_200",
            "rexnetr_300",
        ]
        num_classes = 10  # Just an example number of classes
        for arch in architectures:
            with self.subTest(architecture=arch):
                created_model = get_rexnet_model(arch, num_classes)
                self.assertIsNotNone(created_model)
                # You may want to add more specific tests here to ensure correctness
                # For instance, checking if the returned model is an instance
                # of torch.nn.Module

    def test_unknown_architecture(self):
        """
        Test if the function raises a ValueError for an unknown Rexnet architecture.
        """
        unknown_architecture = "unknown_architecture"
        num_classes = 10
        with self.assertRaises(ValueError):
            get_rexnet_model(unknown_architecture, num_classes)
