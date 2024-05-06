import os
from django.test import TestCase
from fl_common.models.convnext import get_convnext_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingConvNextTestCase(TestCase):
    """
    Test case class for processing Canvnext models.
    """

    def test_convnext_model(self):
        """
        Test case for creating ConvNeXt models.

        Iterates through different ConvNeXt architectures and checks if the created model is an instance
        of `nn.Module`.

        Raises:
            AssertionError: If the assertion fails.
        """
        convnext_types = ['ConvNeXt_Tiny', 'ConvNeXt_Small', 'ConvNeXt_Base', 'ConvNeXt_Large', 'convnext_atto',
                          'convnext_atto_ols', 'convnext_femto', 'convnext_femto_ols', 'convnext_pico',
                          'convnext_pico_ols', 'convnext_nano', 'convnext_nano_ols', 'convnext_tiny_hnf',
                          'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large', 'convnext_large_mlp',
                          'convnext_xlarge', 'convnext_xxlarge', 'convnextv2_atto', 'convnextv2_femto',
                          'convnextv2_pic', 'convnextv2_nano', 'convnextv2_tiny', 'convnextv2_small', 'convnextv2_base',
                          'convnextv2_large', 'convnextv2_huge']
        num_classes = 10  # You can adjust the number of classes as needed

        for convnext_type in convnext_types:
            with self.subTest(convnext_type=convnext_type):
                model = get_convnext_model(convnext_type, num_classes)
                try:
                    model = get_convnext_model(convnext_type, num_classes)
                    self.assertIsNotNone(model)
                except ValueError:
                    self.fail(f"{convnext_type} should be a known Convnext architecture.")

    def test_convnext_unknown_architecture(self):
        """
        Test case for handling unknown ConvNeXt architecture.

        Raises:
            ValueError: If an unknown ConvNeXt architecture is encountered.
            AssertionError: If the assertion fails.
        """
        convnext_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            get_convnext_model(convnext_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown ConvNeXt Architecture : {convnext_type}'
        )
