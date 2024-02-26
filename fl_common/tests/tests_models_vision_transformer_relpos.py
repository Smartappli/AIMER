import os
from django.test import TestCase
from fl_common.models.vision_transformer_relpos import get_vision_transformer_relpos_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingVisionTransformerRelposTestCase(TestCase):
    """
    Test case class for the get_vision_transformer_relpos_model function.
    """

    def test_vision_transformer_relpos_models(self):
        """
        Test all Vision Transformer Relative Position models.
        """
        num_classes = 10
        model_types = ['vit_relpos_base_patch32_plus_rpn_256', 'vit_relpos_base_patch16_plus_240', 'vit_relpos_small_patch16_224',
                       'vit_relpos_medium_patch16_224', 'vit_relpos_base_patch16_224', 'vit_srelpos_small_patch16_224',
                       'vit_srelpos_medium_patch16_224', 'vit_relpos_medium_patch16_cls_224', 'vit_relpos_base_patch16_cls_224',
                       'vit_relpos_base_patch16_clsgap_224', 'vit_relpos_small_patch16_rpn_224', 'vit_relpos_medium_patch16_rpn_224',
                       'vit_relpos_base_patch16_rpn_224']
        for model_type in model_types:
            with self.subTest(model_type=model_type):
                model = get_vision_transformer_relpos_model(model_type, num_classes)
                self.assertIsNotNone(model)
                # Add more specific tests if needed

    def test_invalid_type(self):
        """
        Test for getting an invalid Vision Transformer Relative Position model type.
        """
        num_classes = 10
        with self.assertRaises(ValueError):
            model = get_vision_transformer_relpos_model('invalid_type', num_classes)
            # Ensure it raises ValueError for an unknown vision_transformer_relpos_type
