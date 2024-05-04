import os
from django.test import TestCase
from fl_common.models.efficientvit_mit import get_efficientvit_mit_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingEfficientvit_mitTestCase(TestCase):
    """
    Test case class for processing Efficientvit_mit models.
    """

    def test_all_efficientvit_mit_models(self):
        """Test all Efficientvit_mit models"""
        num_classes = 10
        model_types = ['efficientvit_b0', 'efficientvit_b1', 'efficientvit_b2', 'efficientvit_b3', 'efficientvit_l1',
                       'efficientvit_l2', 'efficientvit_l3']
        for model_type in model_types:
            with self.subTest(model_type=model_type):
                model = get_efficientvit_mit_model(model_type, num_classes)
                self.assertIsNotNone(model)
                # Add more specific tests if needed

    def test_efficientvit_mit_unknown_architecture(self):
        """
        Test case for handling unknown Efficientvit_mit architecture in get_efficientvit_mit_model function.

        Raises:
            AssertionError: If the assertion fails.
            ValueError: If an unknown Efficientvit_mit architecture is provided.
        """
        model_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            # Attempt to get a Vision Transformer model with an unknown architecture
            get_efficientvit_mit_model(model_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown Efficientvit_mit Architecture: {model_type}'
        )
