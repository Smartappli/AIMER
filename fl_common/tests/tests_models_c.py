import os
import torch.nn as nn
from django.test import TestCase
from fl_common.models.cait import get_cait_model
from fl_common.models.coat import get_coat_model
from fl_common.models.convmixer import get_convmixer_model
from fl_common.models.convit import get_convit_model
from fl_common.models.convnext import get_convnext_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingPartCTestCase(TestCase):
    """Cait Model Unit Tests"""

    def test_known_cait_types(self):
        """
        Test the get_cait_model function with known CAIT types.

        For each known CAIT type, this test checks if the function returns a non-None CAIT model
        when provided with that type.

        Parameters:
        - None

        Returns:
        - None

        Raises:
        - AssertionError: If the function returns None for any known CAIT type.
        """
        known_types = [
            "cait_xxs24_224", "cait_xxs24_384", "cait_xxs36_224", "cait_xxs36_384",
            "cait_xs24_384", "cait_s24_224", "cait_s24_384", "cait_s36_384",
            "cait_m36_384", "cait_m48_448"
        ]

        for cait_type in known_types:
            with self.subTest(cait_type=cait_type):
                cait_model = get_cait_model(cait_type, num_classes=10)
                self.assertIsNotNone(cait_model)

    def test_unknown_cait_type(self):
        """
        Test the get_cait_model function behavior when provided with an unknown CAIT type.

        This test checks if the function raises a ValueError when it is called with an unknown CAIT type.

        Parameters:
        - None

        Returns:
        - None

        Raises:
        - AssertionError: If the function does not raise a ValueError for an unknown CAIT type.
        """
        unknown_type = "unknown_cait_type"

        with self.assertRaises(ValueError):
            get_cait_model(unknown_type, num_classes=10)

    """Coat Models Unit Tests"""
    def test_known_coat_types(self):
        """
        Test the get_coat_model function with known COAT types.

        For each known COAT type, this test checks if the function returns a non-None COAT model
        when provided with that type.

        Parameters:
        - None

        Returns:
        - None

        Raises:
        - AssertionError: If the function returns None for any known COAT type.
        """
        known_types = [
            "coat_tiny", "coat_mini", "coat_small",
            "coat_lite_tiny", "coat_lite_mini", "coat_lite_small",
            "coat_lite_medium", "coat_lite_medium_384"
        ]

        for coat_type in known_types:
            with self.subTest(coat_type=coat_type):
                coat_model = get_coat_model(coat_type, num_classes=10)
                self.assertIsNotNone(coat_model)

    def test_unknown_coat_type(self):
        """
        Test the get_coat_model function behavior when provided with an unknown COAT type.

        This test checks if the function raises a ValueError when it is called with an unknown COAT type.

        Parameters:
        - None

        Returns:
        - None

        Raises:
        - AssertionError: If the function does not raise a ValueError for an unknown COAT type.
        """
        unknown_type = "unknown_coat_type"

        with self.assertRaises(ValueError):
            get_coat_model(unknown_type, num_classes=10)

    """Convit Models Unit Tests"""

    def test_valid_architecture(self):
        """Test for a valid Convit architecture."""
        architectures = [
            "convit_tiny",
            "convit_small",
            "convit_base",
        ]
        num_classes = 1000  # Change this to the appropriate number of classes

        for convit_type in architectures:
            with self.subTest(convit_type=convit_type):
                result = get_convit_model(convit_type, num_classes)
                self.assertIsInstance(result, nn.Module)

    def test_unknown_architecture(self):
        """Test for an unknown Convit architecture."""
        convit_type = "unknown_architecture"
        num_classes = 1000  # Change this to the appropriate number of classes
        with self.assertRaises(ValueError):
            get_convit_model(convit_type, num_classes)

    def test_valid_architecture_custom_classes(self):
        """Test for a valid Convit architecture with a custom number of classes."""
        architectures = [
            "convit_tiny",
            "convit_small",
            "convit_base",
        ]
        num_classes = 500  # Change this to a custom number of classes

        for convit_type in architectures:
            with self.subTest(convit_type=convit_type):
                result = get_convit_model(convit_type, num_classes)
                self.assertIsInstance(result, nn.Module)

    """ConvNeXt Model Unit Test"""
    def test_convnet_model(self):
        """
        Test case for creating ConvNeXt models.

        Iterates through different ConvNeXt architectures and checks if the created model is an instance
        of `nn.Module`.

        Raises:
            AssertionError: If the assertion fails.
        """
        convnext_types = ['ConvNeXt_Tiny', 'ConvNeXt_Small', 'ConvNeXt_Base', 'ConvNeXt_Large']
        num_classes = 10  # You can adjust the number of classes as needed

        for convnext_type in convnext_types:
            with self.subTest(convnext_type=convnext_type):
                model = get_convnext_model(convnext_type, num_classes)
                self.assertIsInstance(model, nn.Module)

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

    def test_convnext_last_layer_adaptation(self):
        """
        Test case for ensuring the last layer adaptation in ConvNeXt models.

        Raises:
            AssertionError: If the assertion fails.
        """
        # Provide a known architecture type
        convnext_type = 'ConvNeXt_Large'
        num_classes = 10

        # Override the last layer with a linear layer for testing purposes
        convnext_model = get_convnext_model(convnext_type, num_classes)
        last_layer = None
        for layer in reversed(convnext_model.classifier):
            if isinstance(layer, nn.Linear):
                last_layer = layer
                break

        self.assertIsNotNone(last_layer)
        self.assertEqual(last_layer.out_features, num_classes)

    # Convmixer model unit tests
    def test_get_convmixer_model(self):
        """
        Unit test for the `get_convmixer_model` function.

        Iterates through different Convmixer architectures and verifies whether the function
        returns a valid model instance for each architecture type.

        Parameters:
            self: The test case object.

        Returns:
            None
        """
        num_classes = 1000  # Example number of classes
        convmixer_types = ["convmixer_1536_20", "convmixer_768_32", "convmixer_1024_20_ks9_p14"]

        for convmixer_type in convmixer_types:
            with self.subTest(convmixer_type=convmixer_type):
                model = get_convmixer_model(convmixer_type, num_classes)
                self.assertIsNotNone(model)
                self.assertIsInstance(model, nn.Module)
