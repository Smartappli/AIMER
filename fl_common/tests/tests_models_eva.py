import os
from django.test import TestCase
from fl_common.models.eva import get_eva_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingEvaTestCase(TestCase):
    """Eva Model Unit Tests"""
    def test_known_eva_types(self):
        """
        Test for known Eva architecture types to ensure they return a model without raising any exceptions.
        """
        known_eva_types = [
            'eva_giant_patch14_224',
            'eva_giant_patch14_336',
            'eva_giant_patch14_560',
            'eva02_tiny_patch14_224',
            'eva02_small_patch14_224',
            'eva02_base_patch14_224',
            'eva02_large_patch14_224',
            'eva02_tiny_patch14_336',
            'eva02_small_patch14_336',
            'eva02_base_patch14_448',
            'eva02_large_patch14_448',
            'eva_giant_patch14_clip_224',
            'eva02_base_patch16_clip_224',
            'eva02_large_patch14_clip_224',
            'eva02_large_patch14_clip_336',
            'eva02_enormous_patch14_clip_224'
        ]

        num_classes = 1000  # Assuming 1000 classes for the test

        for eva_type in known_eva_types:
            with self.subTest(eva_type=eva_type):
                try:
                    model = get_eva_model(eva_type, num_classes)
                    self.assertIsNotNone(model)
                except ValueError:
                    self.fail(f"{eva_type} should be a known Eva architecture.")

    def test_unknown_eva_type(self):
        """
        Test for an unknown Eva architecture type to ensure it raises a ValueError.
        """
        unknown_eva_type = "unknown_eva_type"
        num_classes = 1000

        with self.assertRaises(ValueError):
            get_eva_model(unknown_eva_type, num_classes)
