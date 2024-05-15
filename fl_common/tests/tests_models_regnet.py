import os
from django.test import TestCase
from fl_common.models.regnet import get_regnet_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingRegnetTestCase(TestCase):
    """
    Test case class for processing Regnet models.
    """

    def test_get_regnet_model(self):
        """
        Test case for obtaining various RegNet models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # List of RegNet model types to test
        regnet_types = ['RegNet_Y_400MF', 'RegNet_Y_800MF', 'RegNet_Y_1_6GF', 'RegNet_Y_3_2GF', 'RegNet_Y_16GF',
                        'regnetx_002', 'regnetx_004', 'regnetx_004_tv', 'regnetx_006', 'regnetx_008', 'regnetx_016',
                        'regnetx_032', 'regnetx_040', 'regnetx_064', 'regnetx_080', 'regnetx_120', 'regnetx_160',
                        'regnetx_320', 'regnety_002', 'regnety_004', 'regnety_006', 'regnety_008', 'regnety_008_tv',
                        'regnety_016', 'regnety_032', 'regnety_040', 'regnety_064', 'regnety_080', 'regnety_080_tv',
                        'regnety_120', 'regnety_160', 'regnety_320', 'regnety_640', 'regnety_1280', 'regnety_2560',
                        'regnety_040_sgn', 'regnetv_040', 'regnetv_064', 'regnetz_005', 'regnetz_040', 'regnetz_040_h']
        num_classes = 10  # You can adjust the number of classes as needed

        # Loop through each RegNet model type
        for regnet_type in regnet_types:
            with self.subTest(regnet_type=regnet_type):
                # Get the RegNet model for testing
                try:
                    model = get_regnet_model(regnet_type, num_classes)
                    self.assertIsNotNone(model)
                except ValueError:
                    self.fail(f"{regnet_type} should be a known Regnet architecture.")

    def test_regnet_unknown_architecture(self):
        """
        Test case for handling unknown RegNet architecture in get_regnet_model function.

        Raises:
            AssertionError: If the assertion fails.
            ValueError: If an unknown RegNet architecture is provided.
        """
        regnet_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            # Attempt to get a RegNet model with an unknown architecture
            get_regnet_model(regnet_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown RegNet Architecture: {regnet_type}'
        )
