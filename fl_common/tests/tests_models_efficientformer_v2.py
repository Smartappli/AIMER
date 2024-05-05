import os
from django.test import TestCase
from fl_common.models.efficientformer_v2 import get_efficientformer_v2_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingEfficientformerv2TestCase(TestCase):
    """
    Test case class for processing Efficientformer models.
    """

    def test_all_efficientformer_v2_models(self):
        """Test all Efficientformer v2 models"""
        num_classes = 10
        model_types = ['efficientformerv2_s0', 'efficientformerv2_s1', 'efficientformerv2_s2', 'efficientformerv2_l']
        for model_type in model_types:
            with self.subTest(model_type=model_type):
                model = get_efficientformer_v2_model(model_type, num_classes)
                self.assertIsNotNone(model)
                # Add more specific tests if needed

    def test_efficientformer_v2_unknown_architecture(self):
        """
        Test case for handling unknown Efficientformer v2 architecture in get_efficientformer_v2_model function.

        Raises:
            AssertionError: If the assertion fails.
            ValueError: If an unknown Efficientformer v2 architecture is provided.
        """
        model_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            # Attempt to get a Vision Transformer model with an unknown architecture
            get_efficientformer_v2_model(model_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown Efficientformer v2 Architecture: {model_type}'
        )
