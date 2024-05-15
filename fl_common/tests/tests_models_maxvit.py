import os
import torch.nn as nn
from django.test import TestCase
from fl_common.models.maxvit import get_maxvit_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingMaxvitTestCase(TestCase):
    """
    Test case class for processing Maxvit models.
    """

    def test_get_maxvit_model(self):
        """
        Test case for obtaining MaxVit models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # List of MaxVit model types to test
        maxvit_types = [
            'MaxVit_T',
            'coatnet_pico_rw_224',
            'coatnet_nano_rw_224',
            'coatnet_0_rw_224',
            'coatnet_1_rw_224',
            'coatnet_2_rw_224',
            'coatnet_3_rw_224',
            'coatnet_bn_0_rw_224',
            'coatnet_rmlp_nano_rw_224',
            'coatnet_rmlp_0_rw_224',
            'coatnet_rmlp_1_rw_224',
            'coatnet_rmlp_1_rw2_224',
            'coatnet_rmlp_2_rw_224',
            'coatnet_rmlp_2_rw_384',
            'coatnet_rmlp_3_rw_224',
            'coatnet_nano_cc_224',
            'coatnext_nano_rw_224',
            'coatnet_0_224',
            'coatnet_1_224',
            'coatnet_2_224',
            'coatnet_3_224',
            'coatnet_4_224',
            'coatnet_5_224',
            'maxvit_pico_rw_256',
            'maxvit_nano_rw_256',
            'maxvit_tiny_rw_224',
            'maxvit_tiny_rw_256',
            'maxvit_rmlp_pico_rw_256',
            'maxvit_rmlp_nano_rw_256',
            'maxvit_rmlp_tiny_rw_256',
            'maxvit_rmlp_small_rw_224',
            'maxvit_rmlp_small_rw_256',
            "maxvit_rmlp_base_rw_224",
            'maxvit_rmlp_base_rw_384',
            'maxvit_tiny_pm_256',
            'maxxvit_rmlp_nano_rw_256',
            'maxxvit_rmlp_tiny_rw_256',
            'maxxvit_rmlp_small_rw_256',
            'maxxvitv2_nano_rw_256',
            'maxxvitv2_rmlp_base_rw_224',
            'maxxvitv2_rmlp_base_rw_384',
            'maxxvitv2_rmlp_large_rw_224',
            'maxvit_tiny_tf_224',
            'maxvit_tiny_tf_384',
            'maxvit_tiny_tf_512',
            'maxvit_small_tf_224',
            'maxvit_small_tf_384',
            'maxvit_small_tf_512',
            'maxvit_base_tf_224',
            'maxvit_base_tf_384',
            'maxvit_base_tf_512',
            'maxvit_large_tf_224',
            'maxvit_large_tf_384',
            'maxvit_large_tf_512',
            'maxvit_xlarge_tf_224',
            'maxvit_xlarge_tf_384',
            'maxvit_xlarge_tf_512']
        num_classes = 10  # You can adjust the number of classes as needed

        # Loop through each MaxVit model type
        for maxvit_type in maxvit_types:
            with self.subTest(maxvit_type=maxvit_type):
                # Get the MaxVit model for testing
                model = get_maxvit_model(maxvit_type, num_classes)
                # Check if the model is an instance of torch.nn.Module
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_maxvit_unknown_architecture(self):
        """
        Test case for handling unknown MaxVit architecture in get_maxvit_model function.

        Raises:
            AssertionError: If the assertion fails.
            ValueError: If an unknown MaxVit architecture is provided.
        """
        maxvit_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            # Attempt to get a MaxVit model with an unknown architecture
            get_maxvit_model(maxvit_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown MaxVit Architecture: {maxvit_type}'
        )
