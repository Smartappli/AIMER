import os
from django.test import TestCase
from fl_common.models.mvitv2 import get_mvitv2_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingMvitv2TestCase(TestCase):
    """Mvitv2 Model Unit Tests"""
    def test_known_architectures(self):
        architectures = ['mvitv2_tiny', 'mvitv2_small', 'mvitv2_base', 'mvitv2_large',
                         'mvitv2_small_cls', 'mvitv2_base_cls', 'mvitv2_large_cls', 'mvitv2_huge_cls']
        num_classes = 10  # just an example number of classes
        for arch in architectures:
            with self.subTest(architecture=arch):
                created_model = get_mvitv2_model(arch, num_classes)
                self.assertIsNotNone(created_model)
                # You may want to add more specific tests here to ensure correctness
                # For instance, checking if the returned model is an instance of torch.nn.Module

    def test_unknown_architecture(self):
        unknown_architecture = 'unknown_architecture'
        num_classes = 10
        with self.assertRaises(ValueError):
            get_mvitv2_model(unknown_architecture, num_classes)