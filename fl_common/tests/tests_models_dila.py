import os
from django.test import TestCase
from fl_common.models.dila import get_dila_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingDilaTestCase(TestCase):
    """Dila Models Unit Tests"""
    def test_known_dila_types(self):
        known_dila_types = [
            "dla60_res2net",
            "dla60_res2next",
            "dla34",
            "dla46_c",
            "dla46x_c",
            "dla60x_c",
            "dla60",
            "dla60x",
            "dla102",
            "dla102x",
            "dla102x2",
            "dla169"
        ]

        num_classes = 1000  # Assuming 1000 classes for the test

        for dila_type in known_dila_types:
            with self.subTest(dila_type=dila_type):
                try:
                    model = get_dila_model(dila_type, num_classes)
                    self.assertIsNotNone(model)
                except ValueError:
                    self.fail(f"{dila_type} should be a known Dila architecture.")

    def test_unknown_dila_type(self):
        unknown_dila_type = "unknown_dila_type"
        num_classes = 1000

        with self.assertRaises(ValueError):
            get_dila_model(unknown_dila_type, num_classes)
