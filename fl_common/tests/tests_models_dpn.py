import os
from django.test import TestCase
from fl_common.models.dpn import get_dpn_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingDpnTestCase(TestCase):
    """
    Test case class for processing Dpn models.
    """

    def test_all_dpn_models(self):
        """
        Test case to ensure that all DPN (Dual-Path Network) models can be created successfully.

        Iterates through a list of DPN architecture types and attempts to create models for each type.
        Verifies that the models are not None.

        Raises:
            AssertionError: If any DPN model fails to be created.
        """
        dpn_types = [
            "dpn48b",
            "dpn68",
            "dpn68b",
            "dpn92",
            "dpn98",
            "dpn131",
            "dpn107",
        ]
        num_classes = 10  # Example number of classes

        for dpn_type in dpn_types:
            with self.subTest(dpn_type=dpn_type):
                dpn_model = get_dpn_model(dpn_type, num_classes)
                self.assertIsNotNone(dpn_model)

    def test_unknown_dpn_architecture(self):
        """
        Test case to ensure that attempting to create a DPN model with an unknown architecture type raises a ValueError.

        Verifies that a ValueError is raised when attempting to create a DPN model with an unknown architecture type.

        Raises:
            AssertionError: If creating a DPN model with an unknown architecture type does not raise a ValueError.
        """
        unknown_dpn_type = "unknown_dpn"
        num_classes = 10  # Example number of classes

        with self.assertRaises(ValueError):
            get_dpn_model(unknown_dpn_type, num_classes)
