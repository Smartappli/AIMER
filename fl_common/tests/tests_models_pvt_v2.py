import os
from django.test import TestCase
from fl_common.models.pvt_v2 import get_pvt_v2_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingPvt_v2TestCase(TestCase):
    """
    Test case class for processing Pvt v2 models.
    """

    def test_known_pvt_v2_types(self):
        """
        Test for known PVTv2 architecture types to ensure they return a model without raising any exceptions.
        """
        known_pvt_v2_types = [
            "pvt_v2_b0",
            "pvt_v2_b1",
            "pvt_v2_b2",
            "pvt_v2_b3",
            "pvt_v2_b4",
            "pvt_v2_b5",
            "pvt_v2_b2_li",
        ]
        num_classes = 1000  # Assuming 1000 classes for the test

        for pvt_v2_type in known_pvt_v2_types:
            with self.subTest(pvt_v2_type=pvt_v2_type):
                try:
                    model = get_pvt_v2_model(pvt_v2_type, num_classes)
                    self.assertIsNotNone(model)
                except ValueError:
                    self.fail(f"{pvt_v2_type} should be a known PVTv2 architecture.")

    def test_unknown_pvt_v2_type(self):
        """
        Test to ensure that an unknown PVTv2 architecture type raises a ValueError.
        """
        unknown_pvt_v2_type = "unknown_pvt_v2_type"
        num_classes = 1000

        with self.assertRaises(ValueError):
            get_pvt_v2_model(unknown_pvt_v2_type, num_classes)
