import os
from django.test import TestCase
from fl_common.models.swin_transformer_v2_cr import get_swin_transformer_v2_cr_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingSwinTransformerv2crTestCase(TestCase):

    """
    Test case class for processing Swin Transformer models.
    """

    def test_get_swin_model(self):
        """
        Test case for obtaining various Swin Transformer v2 cr models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # List of Swin Transformer model types to test
        swin_types = ["swinv2_cr_tiny_384", "swinv2_cr_tiny_224", "swinv2_cr_tiny_ns_224", "swinv2_cr_small_384",
                      "swinv2_cr_small_224", "swinv2_cr_small_ns_224", "swinv2_cr_small_ns_256", "swinv2_cr_base_384"]

        num_classes = 10  # You can adjust the number of classes as needed

        # Loop through each Swin Transformer model type
        for swin_type in swin_types:
            with self.subTest(swin_type=swin_type):
                # Get the Swin Transformer model for testing
                model = get_swin_transformer_v2_cr_model(swin_type, num_classes)
                # Check if the model is an instance of torch.nn.Module
                self.assertIsNotNone(model)

    def test_swin_unknown_architecture(self):
        """
        Test case for handling unknown Swin Transformer v2 cr architecture in get_swin_transformer_v2_cr_model function.

        Raises:
            AssertionError: If the assertion fails.
            ValueError: If an unknown Swin Transformer architecture is provided.
        """
        swin_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            # Attempt to get a Swin Transformer model with an unknown architecture
            get_swin_transformer_v2_cr_model(swin_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown Swin Transformer v2 cr Architecture: {swin_type}'
        )
