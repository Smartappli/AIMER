import os
from django.test import TestCase
from fl_common.models.vision_transformer_hybrid import (
    get_vision_transformer_hybrid_model,
)

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingVisionTransformerHybridTestCase(TestCase):
    """
    Test case class for processing Vision Transformer Hybrid models.
    """

    def test_vision_transformer_hybrid_models(self):
        """Test all Vision Transformer Hybrid models"""
        num_classes = 10
        model_types = [
            "vit_tiny_r_s16_p8_224",
            "vit_tiny_r_s16_p8_384",
            "vit_small_r26_s32_224",
            "vit_small_r26_s32_384",
            "vit_base_r26_s32_224",
            "vit_base_r50_s16_224",
            "vit_base_r50_s16_384",
            "vit_large_r50_s32_224",
            "vit_large_r50_s32_384",
            "vit_small_resnet26d_224",
            "vit_small_resnet50d_s16_224",
            "vit_base_resnet26d_224",
            "vit_base_resnet50d_224",
        ]
        for model_type in model_types:
            with self.subTest(model_type=model_type):
                model = get_vision_transformer_hybrid_model(
                    model_type,
                    num_classes,
                )
                self.assertIsNotNone(model)
                # Add more specific tests if needed

    def test_vision_transformer_hybrid_unknown_architecture(self):
        """
        Test case for handling unknown Vision Transformer Hybrid architecture in get_vision_transformer_hybrid
        model function.

        Raises:
            AssertionError: If the assertion fails.
            ValueError: If an unknown Vision Transformer Hybrid architecture is provided.
        """
        vision_type = "UnknownArchitecture"
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            # Attempt to get a Vision Transformer model with an unknown
            # architecture
            get_vision_transformer_hybrid_model(vision_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f"Unknown Vision Transformer Hybrid Architecture: {vision_type}",
        )
