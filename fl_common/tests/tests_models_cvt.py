import os
from django.test import TestCase
from fl_common.models.cvt import get_cvt_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingCvtTestCase(TestCase):
    """²²
    Test case class for processing cvt models.
    """

    def test_known_cvt_types(self):
        """
        Test if the function returns a valid cvt model for known cvt types.
        """
        cvt_types = [
            "cvt_13",
            "cvt_21",
            "cvt_w24",
        ]
        num_classes = 10

        for cvt_type in cvt_types:
            cvt_model = get_cvt_model(cvt_type, num_classes)
            self.assertIsNotNone(cvt_model)
            # Check if the model has the expected number of output classes
            self.assertEqual(cvt_model.num_classes, num_classes)

    def test_unknown_cvt_type(self):
        """
        Test if the function raises a ValueError for an unknown cvt type.
        """
        unknown_cvt_type = "unknown_cvt_type"
        num_classes = 10

        with self.assertRaises(ValueError):
            get_cvt_model(unknown_cvt_type, num_classes)
