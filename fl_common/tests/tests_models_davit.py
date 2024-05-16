import os
from django.test import TestCase
from fl_common.models.davit import get_davit_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingDavitTestCase(TestCase):
    """
    Test case class for processing Davit models.
    """

    def test_known_davit_types(self):
        """
        Test if the function returns a valid Davit model for known Davit types.
        """
        davit_types = [
            "davit_tiny",
            "davit_small",
            "davit_base",
            "davit_large",
            "davit_huge",
            "davit_giant",
        ]
        num_classes = 10

        for davit_type in davit_types:
            davit_model = get_davit_model(davit_type, num_classes)
            self.assertIsNotNone(davit_model)
            # Check if the model has the expected number of output classes
            self.assertEqual(davit_model.num_classes, num_classes)

    def test_unknown_davit_type(self):
        """
        Test if the function raises a ValueError for an unknown Davit type.
        """
        unknown_davit_type = "unknown_davit_type"
        num_classes = 10

        with self.assertRaises(ValueError):
            get_davit_model(unknown_davit_type, num_classes)
