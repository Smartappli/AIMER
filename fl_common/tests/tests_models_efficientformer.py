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

    def test_invalid_type(self):
        """Test for getting an invalid Efficientformer model type"""
        num_classes = 10
        with self.assertRaises(ValueError):
            model = get_efficientformer_model('invalid_type', num_classes)
            # Ensure it raises ValueError for an unknown efficientformer_type
