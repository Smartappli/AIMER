import os

from django.test import TestCase
from torch import nn

from fl_common.models.volo import get_volo_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingVoloTestCase(TestCase):
    """
    Test case class for processing Volo models.
    """

    def test_all_volo_models(self):
        """
        Test case for obtaining various Volo models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        volo_types = [
            "volo_d1_224",
            "volo_d1_384",
            "volo_d2_224",
            "volo_d2_384",
            "volo_d3_224",
            "volo_d3_448",
            "volo_d4_224",
            "volo_d4_448",
            "volo_d5_224",
            "volo_d5_448",
            "volo_d5_512",
        ]

        for volo_type in volo_types:
            with self.subTest(volo_type=volo_type):
                # Get the Volo model for testing
                model = get_volo_model(volo_type, num_classes=10)
                # Check if the model is an instance of torch.nn.Module
                self.assertTrue(isinstance(model, nn.Module))

    def test_volo_unknown_type(self):
        """
        Test case for handling unknown Volo model type in get_volo_model function.

        Raises:
            AssertionError: If the assertion fails.
            ValueError: If an unknown Volo model type is provided.
        """
        with self.assertRaises(ValueError):
            # Attempt to get a Volo model with an unknown type
            get_volo_model("unknown_type", num_classes=10)
