import os
from django.test import TestCase
from fl_common.models.repvit import get_repvit_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingRepvitTestCase(TestCase):
    """
    Test case class for processing Regvit models.
    """

    def test_known_repvit_type(self):
        """
        Test if the function returns a valid RepVIT model for known RepVIT types.
        """
        repvit_types = [
            "repvit_m1",
            "repvit_m2",
            "repvit_m3",
            "repvit_m0_9",
            "repvit_m1_0",
            "repvit_m1_1",
            "repvit_m1_5",
            "repvit_m2_3",
        ]
        num_classes = 10  # Just for testing purposes

        for repvit_type in repvit_types:
            repvit_model = get_repvit_model(repvit_type, num_classes)
            self.assertIsNotNone(repvit_model)
            # Check if the model has the expected number of output classes
            self.assertEqual(repvit_model.num_classes, num_classes)

    def test_unknown_repvit_type(self):
        """
        Test if the function raises a ValueError for an unknown RepVIT type.
        """
        unknown_repvit_type = "unknown_repvit_type"
        num_classes = 10

        with self.assertRaises(ValueError):
            get_repvit_model(unknown_repvit_type, num_classes)
