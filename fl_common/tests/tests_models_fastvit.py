import os

from django.test import TestCase
from fl_common.models.fastvit import get_fastvit_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingFastVitTestCase(TestCase):
    """
    Test case class for processing Fastvit models.
    """

    def test_all_architectures(self):
        """
        Test the creation of FastViT models for all specified architectures.

        Iterates through the specified architectures, creates the corresponding models,
        and checks that each model is not None.

        Parameters:
        - None

        Returns:
        - None
        """
        architectures = [
            "fastvit_t8",
            "fastvit_t12",
            "fastvit_s12",
            "fastvit_sa12",
            "fastvit_sa24",
            "fastvit_sa36",
            "fastvit_ma36",
        ]

        num_classes = 10  # You can adjust this based on your use case

        for architecture in architectures:
            with self.subTest(architecture=architecture):
                model = get_fastvit_model(architecture, num_classes)
                self.assertIsNotNone(model)

    def test_unknown_architecture(self):
        """
        Test that a ValueError is raised for an unknown architecture.

        Parameters:
        - None

        Returns:
        - None
        """
        with self.assertRaises(ValueError):
            get_fastvit_model("unknown_architecture", num_classes=10)
