import os
from django.test import TestCase
from fl_common.models.mlp_mixer import get_mlp_mixer_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingMlpMixerTestCase(TestCase):
    """MlpMixer Model Unit Tests"""
    def test_known_mlp_mixer_types(self):
        """
        Test for known Mlp Mixer architecture types to ensure they return a model without raising any exceptions.
        """
        known_mlp_mixer_types = ['mixer_s32_224', 'mixer_s16_224', 'mixer_b32_224', 'mixer_b16_224', 'mixer_l32_224',
                                 'mixer_l16_224', 'gmixer_12_224', 'gmixer_24_224', 'resmlp_12_224', 'resmlp_24_224',
                                 'resmlp_36_224', 'resmlp_big_24_224', 'gmlp_ti16_224', 'gmlp_s16_224', 'gmlp_b16_224']
        num_classes = 1000  # Assuming 1000 classes for the test

        for mlp_mixer_type in known_mlp_mixer_types:
            with self.subTest(mlp_mixer_type=mlp_mixer_type):
                try:
                    model = get_mlp_mixer_model(mlp_mixer_type, num_classes)
                    self.assertIsNotNone(model)
                except ValueError:
                    self.fail(f"{mlp_mixer_type} should be a known Mlp Mixer architecture.")

    def test_unknown_mlp_mixer_type(self):
        """
        Test to ensure that an unknown Mlp Mixer architecture type raises a ValueError.
        """
        unknown_mlp_mixer_type = "unknown_mlp_mixer_type"
        num_classes = 1000

        with self.assertRaises(ValueError):
            get_mlp_mixer_model(unknown_mlp_mixer_type, num_classes)
