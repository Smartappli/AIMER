import os
from django.test import TestCase
from fl_common.models.metaformer import get_metaformer_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingMetaformerTestCase(TestCase):
    """
    Test case class for processing Metaformer models.
    """

    def test_known_metaformer_types(self):
        """
        Test for known Metaformer architecture types to ensure they return a model without raising any exceptions.
        """
        known_metaformer_types = [
            "poolformer_s12",
            "poolformer_s24",
            "poolformer_s36",
            "poolformer_m36",
            "poolformer_m48",
            "poolformerv2_s12",
            "poolformerv2_s24",
            "poolformerv2_s36",
            "poolformerv2_m36",
            "poolformerv2_m48",
            "convformer_s18",
            "convformer_s36",
            "convformer_m36",
            "convformer_b36",
            "caformer_s18",
            "caformer_s36",
            "caformer_m36",
            "caformer_b36",
        ]
        num_classes = 1000  # Assuming 1000 classes for the test

        for metaformer_type in known_metaformer_types:
            with self.subTest(metaformer_type=metaformer_type):
                try:
                    model = get_metaformer_model(metaformer_type, num_classes)
                    self.assertIsNotNone(model)
                except ValueError:
                    self.fail(
                        f"{metaformer_type} should be a known Metaformer architecture."
                    )

    def test_unknown_metaformer_type(self):
        """
        Test to ensure that an unknown Metaformer architecture type raises a ValueError.
        """
        unknown_metaformer_type = "unknown_metaformer_type"
        num_classes = 1000

        with self.assertRaises(ValueError):
            get_metaformer_model(unknown_metaformer_type, num_classes)
