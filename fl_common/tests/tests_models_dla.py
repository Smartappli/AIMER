import os
from django.test import TestCase
from fl_common.models.dla import get_dla_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingDlaTestCase(TestCase):
    """
    Test case class for processing Dla models.
    """

    def test_known_dila_types(self):
        """
        Test case to ensure that known Dila architectures can be created.

        Iterates through a list of known Dila architecture types and attempts to create models for each type.
        Verifies that the models are not None.

        Raises:
            AssertionError: If any known Dila architecture fails to be created or if any architecture is unknown.
        """
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
            "dla169",
        ]

        num_classes = 1000  # Assuming 1000 classes for the test

        for dila_type in known_dila_types:
            with self.subTest(dila_type=dila_type):
                try:
                    model = get_dla_model(dila_type, num_classes)
                    self.assertIsNotNone(model)
                except ValueError:
                    self.fail(
                        f"{dila_type} should be a known Dila architecture.",
                    )

    def test_unknown_dila_type(self):
        """
        Test case to ensure that attempting to create a Dila model with an unknown architecture type raises a ValueError.

        Verifies that a ValueError is raised when attempting to create a Dila model with an unknown architecture type.

        Raises:
            AssertionError: If creating a Dila model with an unknown architecture type does not raise a ValueError.
        """
        unknown_dila_type = "unknown_dila_type"
        num_classes = 1000

        with self.assertRaises(ValueError):
            get_dla_model(unknown_dila_type, num_classes)
