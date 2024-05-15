import os
from django.test import TestCase
from fl_common.models.vovnet import get_vovnet_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingVovnetTestCase(TestCase):
    """
    Test case class for processing Vovnet models.
    """

    def test_known_vovnet_architecture(self):
        """
        Test the function with known architectures.

        For each known architecture, it checks that no exceptions are raised,
        and the returned model is not None.

        Supported architectures:
        - 'vovnet39a'
        - 'vovnet57a'
        - 'ese_vovnet19b_slim_dw'
        - 'ese_vovnet19b_dw'
        - 'ese_vovnet19b_slim'
        - 'ese_vovnet39b'
        - 'ese_vovnet57b'
        - 'ese_vovnet99b'
        - 'eca_vovnet39b'
        - 'ese_vovnet39b_evos'
        """
        architectures = [
            'vovnet39a',
            'vovnet57a',
            'ese_vovnet19b_slim_dw',
            'ese_vovnet19b_dw',
            'ese_vovnet19b_slim',
            'ese_vovnet39b',
            'ese_vovnet57b',
            'ese_vovnet99b',
            'eca_vovnet39b',
            'ese_vovnet39b_evos']

        for arch in architectures:
            with self.subTest(architecture=arch):
                # Ensure no exceptions are raised
                model = get_vovnet_model(arch, num_classes=10)
                self.assertIsNotNone(model)

    def test_unknown_vovnet_architecture(self):
        """
        Test the function with known architectures.
        Ensure that no exceptions are raised, and the returned model is not None.
        """
        # Test the function with an unknown architecture
        with self.assertRaises(ValueError):
            get_vovnet_model('unknown_architecture', num_classes=10)
