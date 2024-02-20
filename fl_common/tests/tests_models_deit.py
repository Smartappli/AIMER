import os
from django.test import TestCase
from fl_common.models.deit import get_deit_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingDeitTestCase(TestCase):
    """Det Models Unit Tests"""
    def test_known_deit_types(self):
        known_deit_types = [
            "deit_tiny_patch16_224",
            "deit_small_patch16_224",
            "deit_base_patch16_224",
            "deit_base_patch16_384",
            "deit_tiny_distilled_patch16_224",
            "deit_small_distilled_patch16_224",
            "deit_base_distilled_patch16_224",
            "deit_base_distilled_patch16_384",
            "deit3_small_patch16_224",
            "deit3_small_patch16_384",
            "deit3_medium_patch16_224",
            "deit3_base_patch16_224",
            "deit3_base_patch16_384",
            "deit3_large_patch16_224",
            "deit3_large_patch16_384",
            "deit3_huge_patch14_224"
        ]

        num_classes = 1000  # Assuming 1000 classes for the test

        for deit_type in known_deit_types:
            with self.subTest(deit_type=deit_type):
                try:
                    model = get_deit_model(deit_type, num_classes)
                    self.assertIsNotNone(model)
                except ValueError:
                    self.fail(f"{deit_type} should be a known Deit architecture.")

    def test_unknown_deit_type(self):
        unknown_deit_type = "unknown_deit_type"
        num_classes = 1000

        with self.assertRaises(ValueError):
            get_deit_model(unknown_deit_type, num_classes)
