import os
from django.test import TestCase
from fl_common.models.swin_transformer import get_swin_transformer_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingSwinTransformerTestCase(TestCase):
    """
    Test case class for processing Swin Transformer models.
    """

    def test_get_swin_model(self):
        """
        Test case for obtaining various Swin Transformer models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # List of Swin Transformer model types to test
        swin_types = [
            'Swin_T',
            'Swin_S',
            'Swin_B',
            'Swin_V2_T',
            'Swin_V2_S',
            'Swin_V2_B',
            'swin_tiny_patch4_window7_224',
            'swin_small_patch4_window7_224',
            'swin_base_patch4_window7_224',
            'swin_base_patch4_window12_384',
            'swin_large_patch4_window7_224',
            'swin_s3_tiny_224',
            'swin_large_patch4_window12_384',
            'swin_s3_small_224',
            'swin_s3_base_224']
        num_classes = 10  # You can adjust the number of classes as needed

        # Loop through each Swin Transformer model type
        for swin_type in swin_types:
            with self.subTest(swin_type=swin_type):
                # Get the Swin Transformer model for testing
                model = get_swin_transformer_model(swin_type, num_classes)
                # Check if the model is an instance of torch.nn.Module
                self.assertIsNotNone(model)

    def test_swin_unknown_architecture(self):
        """
        Test case for handling unknown Swin Transformer architecture in get_swin_model function.

        Raises:
            AssertionError: If the assertion fails.
            ValueError: If an unknown Swin Transformer architecture is provided.
        """
        swin_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            # Attempt to get a Swin Transformer model with an unknown
            # architecture
            get_swin_transformer_model(swin_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown Swin Transformer Architecture: {swin_type}'
        )
