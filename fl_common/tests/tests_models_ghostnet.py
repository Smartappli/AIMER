import os
from django.test import TestCase
from fl_common.models.ghostnet import get_ghostnet_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingGhostnetTestCase(TestCase):
    """
    Test case class for processing Ghostnet models.
    """

    def test_known_ghostnet_types(self):
        """
        Test case to ensure that known GhostNet architectures can be created successfully.

        Iterates through a list of known GhostNet architecture types and attempts to create models for each type.
        Verifies that the models are not None.

        Raises:
            AssertionError: If any known GhostNet architecture fails to be created.
        """
        known_ghostnet_types = [
            "ghostnet_050",
            "ghostnet_100",
            "ghostnet_130",
            "ghostnetv2_100",
            "ghostnetv2_130",
            "ghostnetv2_160",
        ]
        num_classes = 1000  # Example number of classes

        for ghostnet_type in known_ghostnet_types:
            with self.subTest(ghostnet_type=ghostnet_type):
                try:
                    model = get_ghostnet_model(ghostnet_type, num_classes)
                    self.assertIsNotNone(model)
                except ValueError:
                    self.fail(
                        f"{ghostnet_type} should be a known GhostNet architecture."
                    )

    def test_unknown_ghostnet_type(self):
        """
        Test case to ensure that attempting to create a GhostNet model with an unknown architecture type raises a ValueError.

        Verifies that a ValueError is raised when attempting to create a GhostNet model with an unknown architecture type.

        Raises:
            AssertionError: If creating a GhostNet model with an unknown architecture type does not raise a ValueError.
        """
        unknown_ghostnet_type = "unknown_ghostnet_type"
        num_classes = 1000

        with self.assertRaises(ValueError):
            get_ghostnet_model(unknown_ghostnet_type, num_classes)
