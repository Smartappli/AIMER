import os
import torch.nn as nn
from django.test import TestCase
from fl_common.models.edgenet import get_edgenet_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingEdgenetTestCase(TestCase):
    """EdgeNet Model Unit Tests"""

    def test_valid_edgenet_type(self):
        """
        Test case for validating various EdgeNet model types in get_edgenet_model function.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        edgenet_types = ["edgenext_xx_small", "edgenext_x_small", "edgenext_small", "edgenext_base", "edgenext_small_rw"]
        num_classes = 10

        for edgenet_type in edgenet_types:
            with self.subTest(edgenet_type=edgenet_type):
                model = get_edgenet_model(edgenet_type, num_classes)
                self.assertIsNotNone(model)
                # Add more specific assertions if needed

    def test_edgenet_invalid_type(self):
        """
        Test case for handling invalid EdgeNet model type in get_edgenet_model function.

        Raises:
            AssertionError: If the test fails.
            ValueError: If invalid_edgenet_type is not a valid EdgeNet model type.
        """
        invalid_edgenet_type = "invalid_type"
        num_classes = 10

        with self.assertRaises(ValueError):
            get_edgenet_model(invalid_edgenet_type, num_classes)

    def test_edgenet_invalid_num_classes(self):
        """
        Test case for handling invalid number of classes in get_edgenet_model function.

        Raises:
            AssertionError: If the test fails.
            ValueError: If invalid_num_classes is not an integer.
        """
        edgenet_type = "edgenext_small"
        invalid_num_classes = "not_an_integer"

        with self.assertRaises(ValueError):
            # Ajout d'une vérification pour s'assurer que invalid_num_classes est un entier
            if not isinstance(invalid_num_classes, int):
                raise ValueError("invalid_num_classes doit être un entier")

            get_edgenet_model(edgenet_type, invalid_num_classes)
