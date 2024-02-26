import os
from django.test import TestCase
from fl_common.models.vision_transformer_sam import get_vision_transformer_sam_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingVisionTransformerSamTestCase(TestCase):
    """
    Test case class for the get_vision_transformer_sam_model function.
    """

    def test_vision_transformer_sam_models(self):
        """
        Test all Vision Transformer SAM models.
        """
        num_classes = 10
        model_types = ['samvit_base_patch16', 'samvit_large_patch16', 'samvit_huge_patch16', 'samvit_base_patch16_224']
        for model_type in model_types:
            with self.subTest(model_type=model_type):
                model = get_vision_transformer_sam_model(model_type, num_classes)
                self.assertIsNotNone(model)
                # Add more specific tests if needed

    def test_invalid_type(self):
        """
        Test for getting an invalid Vision Transformer SAM model type.
        """
        num_classes = 10
        with self.assertRaises(ValueError):
            model = get_vision_transformer_sam_model('invalid_type', num_classes)
            # Ensure it raises ValueError for an unknown vision_transformer_sam_type
