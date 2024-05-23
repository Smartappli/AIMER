import os
from django.test import TestCase
from fl_common.models.efficientvit_msra import get_efficientvit_msra_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingEfficientVit_MsraTestCase(TestCase):
    """
    Test case class for processing Efficientvit_msramodels.
    """

    def test_all_efficientvit_msra_models(self):
        """Test all Efficientvit_msra models"""
        num_classes = 10
        model_types = [
            "efficientvit_m0",
            "efficientvit_m1",
            "efficientvit_m2",
            "efficientvit_m3",
            "efficientvit_m4",
            "efficientvit_m5",
        ]
        for model_type in model_types:
            with self.subTest(model_type=model_type):
                model = get_efficientvit_msra_model(model_type, num_classes)
                self.assertIsNotNone(model)

    def test_efficientvit_msra_unknown_architecture(self):
        """
        Test case for handling unknown Efficientvit_msra architecture in get_efficientvit_msra_model function.

        Raises:
            AssertionError: If the assertion fails.
            ValueError: If an unknown Efficientvit_msra architecture is provided.
        """
        model_type = "UnknownArchitecture"
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            # Attempt to get a Vision Transformer model with an unknown
            # architecture
            get_efficientvit_msra_model(model_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f"Unknown EfficientViT-MSRA Architecture: {model_type}",
        )
