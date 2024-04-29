import os
import torch.nn as nn
from django.test import TestCase
from fl_common.models.vision_transformer import get_vision_transformer_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingVisionTransformerTestCase(TestCase):
    """
    Test case class for processing Vision Transformer models.
    """

    def test_get_vision_model(self):
        """
        Test case for obtaining various Vision Transformer models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # List of Vision Transformer model types to test
        vision_types = ['ViT_B_16', 'ViT_B_32', 'ViT_L_16', 'ViT_L_32', 'ViT_H_14', "vit_tiny_patch16_224",
                        "vit_tiny_patch16_384", "vit_small_patch32_224", "vit_small_patch32_384",
                        "vit_small_patch16_224", "vit_small_patch16_384", "vit_small_patch8_224",
                        "vit_base_patch32_224", "vit_base_patch32_384", "vit_base_patch16_224",
                        "vit_base_patch16_384", "vit_base_patch8_224", "vit_large_patch32_224",
                        "vit_large_patch32_384", "vit_large_patch16_224", "vit_large_patch16_384",
                        "vit_large_patch14_224", "vit_large_patch14_224", "vit_giant_patch14_224",
                        "vit_gigantic_patch14_224", "vit_base_patch16_224_miil", "vit_medium_patch16_gap_240",
                        "vit_medium_patch16_gap_256", "vit_medium_patch16_gap_384", "vit_base_patch16_gap_224",
                        "vit_huge_patch14_gap_224", "vit_huge_patch16_gap_448", "vit_giant_patch16_gap_224",
                        "vit_xsmall_patch16_clip_224", "vit_medium_patch32_clip_224", "vit_medium_patch16_clip_224",
                        "vit_betwixt_patch32_clip_224", "vit_base_patch32_clip_224", "vit_base_patch32_clip_256",
                        "vit_base_patch32_clip_384", "vit_base_patch32_clip_448", "vit_base_patch16_clip_224",
                        "vit_base_patch16_clip_384", "vit_large_patch14_clip_224", "vit_large_patch14_clip_336",
                        "vit_huge_patch14_clip_224", "vit_huge_patch14_clip_336", "vit_huge_patch14_clip_378",
                        "vit_giant_patch14_clip_224", "vit_gigantic_patch14_clip_224",
                        "vit_base_patch32_clip_quickgelu_224", "vit_base_patch16_clip_quickgelu_224",
                        "vit_large_patch14_clip_quickgelu_224", "vit_large_patch14_clip_quickgelu_336",
                        "vit_huge_patch14_clip_quickgelu_224", "vit_huge_patch14_clip_quickgelu_378",
                        "vit_base_patch32_plus_256", "vit_base_patch16_plus_240", "vit_base_patch16_rpn_224",
                        "vit_small_patch16_36x1_224", "vit_small_patch16_18x2_224", "vit_base_patch16_18x2_224",
                        "eva_large_patch14_196", "eva_large_patch14_336", "flexivit_small", "flexivit_base",
                        "flexivit_large", "vit_base_patch16_xp_224", "vit_large_patch14_xp_224",
                        "vit_huge_patch14_xp_224", "vit_small_patch14_dinov2", "vit_base_patch14_dinov2",
                        "vit_large_patch14_dinov2", "vit_giant_patch14_dinov2", "vit_small_patch14_reg4_dinov2",
                        "vit_base_patch14_reg4_dinov2", "vit_large_patch14_reg4_dinov2",
                        "vit_giant_patch14_reg4_dinov2", "vit_base_patch16_siglip_224", "vit_base_patch16_siglip_256",
                        "vit_base_patch16_siglip_384", "vit_base_patch16_siglip_512", "vit_large_patch16_siglip_256",
                        "vit_large_patch16_siglip_384", "vit_so400m_patch14_siglip_224",
                        "vit_so400m_patch14_siglip_384", "vit_medium_patch16_reg4_256",
                        "vit_medium_patch16_reg4_gap_256", "vit_base_patch16_reg4_gap_256",
                        "vit_so150m_patch16_reg4_map_256", "vit_so150m_patch16_reg4_gap_256"]

        num_classes = 10  # You can adjust the number of classes as needed

        # Loop through each Vision Transformer model type
        for vision_type in vision_types:
            with self.subTest(vision_type=vision_type):
                # Get the Vision Transformer model for testing
                model = get_vision_transformer_model(vision_type, num_classes)
                # Check if the model is an instance of torch.nn.Module
                self.assertIsInstance(model, nn.Module, msg=f'get_vision_model {vision_type} KO')

    def test_vision_unknown_architecture(self):
        """
        Test case for handling unknown Vision Transformer architecture in get_vision_model function.

        Raises:
            AssertionError: If the assertion fails.
            ValueError: If an unknown Vision Transformer architecture is provided.
        """
        vision_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            # Attempt to get a Vision Transformer model with an unknown architecture
            get_vision_transformer_model(vision_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown Vision Transformer Architecture: {vision_type}'
        )
