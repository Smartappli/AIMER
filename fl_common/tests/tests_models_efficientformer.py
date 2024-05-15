import os
from django.test import TestCase
from fl_common.models.efficientformer import get_efficientformer_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingEfficientformerTestCase(TestCase):
    """
    Test case class for processing Efficientformer models.
    """

    def test_all_efficientformer_models(self):
        """Test all Efficientformer models"""
        num_classes = 10
        model_types = ['efficientformer_l1', 'efficientformer_l3', 'efficientformer_l7']
        for model_type in model_types:
            with self.subTest(model_type=model_type):
                model = get_efficientformer_model(model_type, num_classes)
                self.assertIsNotNone(model)
                # Add more specific tests if needed

    def test_efficientformer_unknown_architecture(self):
        """
        Test case for handling unknown Efficientformer architecture in get_efficientformer_model function.

        Raises:
            AssertionError: If the assertion fails.
            ValueError: If an unknown Efficientformer architecture is provided.
        """
        model_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            # Attempt to get a Vision Transformer model with an unknown architecture
            get_efficientformer_model(model_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown Efficientformer Architecture: {model_type}'
        )
