import os

from django.test import TestCase

from fl_common.models.nfnet import get_nfnet_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingNfnetTestCase(TestCase):
    """Nfnet Model Unit Tests"""

    def test_known_nfnet_types(self):
        """
        Test for known NFNet architecture types to ensure they return a model without raising any exceptions.
        """
        known_nfnet_types = [
            "dm_nfnet_f0",
            "dm_nfnet_f1",
            "dm_nfnet_f2",
            "dm_nfnet_f3",
            "dm_nfnet_f4",
            "dm_nfnet_f5",
            "dm_nfnet_f6",
            "nfnet_f0",
            "nfnet_f1",
            "nfnet_f2",
            "nfnet_f3",
            "nfnet_f4",
            "nfnet_f5",
            "nfnet_f6",
            "nfnet_f7",
            "nfnet_l0",
            "eca_nfnet_l0",
            "eca_nfnet_l1",
            "eca_nfnet_l2",
            "eca_nfnet_l3",
            "nf_regnet_b0",
            "nf_regnet_b1",
            "nf_regnet_b2",
            "nf_regnet_b3",
            "nf_regnet_b4",
            "nf_regnet_b5",
            "nf_resnet26",
            "nf_resnet50",
            "nf_resnet101",
            "nf_seresnet26",
            "nf_seresnet50",
            "nf_seresnet101",
            "nf_ecaresnet26",
            "nf_ecaresnet50",
            "nf_ecaresnet101",
            "test_nfnet",
        ]
        num_classes = 1000  # Assuming 1000 classes for the test

        for nfnet_type in known_nfnet_types:
            with self.subTest(nfnet_type=nfnet_type):
                try:
                    model = get_nfnet_model(nfnet_type, num_classes)
                    self.assertIsNotNone(model)
                except ValueError:
                    self.fail(
                        f"{nfnet_type} should be a known NFNet architecture.",
                    )

    def test_unknown_nfnet_type(self):
        """
        Test to ensure that an unknown NFNet architecture type raises a ValueError.
        """
        unknown_nfnet_type = "unknown_nfnet_type"
        num_classes = 1000

        with self.assertRaises(ValueError):
            get_nfnet_model(unknown_nfnet_type, num_classes)
