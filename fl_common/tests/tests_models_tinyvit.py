import os

from django.test import TestCase

from fl_common.models.tiny_vit import get_tiny_vit_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingTinyVitTestCase(TestCase):
    """
    Test case class for processing TinyVit models.
    """

    def test_all_tiny_vit_models(self):
        """
        Test the creation of all Tiny Vision Transformer (TinyViT) models.

        Iterates through all valid TinyViT types and checks if the returned model is not None.

        Parameters:
        - None

        Returns:
        - None

        Raises:
        - AssertionError: If any of the TinyViT models is None.
        """
        tiny_vit_types = [
            "tiny_vit_5m_224",
            "tiny_vit_11m_224",
            "tiny_vit_21m_224",
            "tiny_vit_21m_384",
            "tiny_vit_21m_512",
        ]
        num_classes = 10

        for tiny_vit_type in tiny_vit_types:
            with self.subTest(tiny_vit_type=tiny_vit_type):
                tiny_vit_model = get_tiny_vit_model(tiny_vit_type, num_classes)
                self.assertIsNotNone(tiny_vit_model)

    def test_unknown_tiny_vit_type(self):
        """
        Test the behavior of get_tiny_vit_model when an unknown TinyViT type is specified.

        Parameters:
        - None

        Returns:
        - None

        Raises:
        - ValueError: If an unknown TinyViT architecture is specified.
        """
        with self.assertRaises(ValueError):
            get_tiny_vit_model("unknown_type", num_classes=10)
