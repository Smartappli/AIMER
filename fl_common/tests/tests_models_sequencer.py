import os
from django.test import TestCase
from fl_common.models.sequencer import get_sequencer_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingSequencerTestCase(TestCase):
    """
    Test case class for processing Sequencer models.
    """

    def test_known_types(self):
        """
        Test for known sequencer architecture types to ensure they return a model without raising any exceptions.
        """
        known_types = ['sequencer2d_s', 'sequencer2d_m', 'sequencer2d_l']
        num_classes = 1000  # Assuming 1000 classes for the test

        for sequencer_type in known_types:
            with self.subTest(sequencer_type=sequencer_type):
                try:
                    model = get_sequencer_model(sequencer_type, num_classes)
                    self.assertIsNotNone(model)
                except ValueError:
                    self.fail(
                        f"{sequencer_type} should be a known sequencer architecture.")

    def test_unknown_type(self):
        """
        Test to ensure that an unknown sequencer architecture type raises a ValueError.
        """
        unknown_type = "unknown_type"
        num_classes = 1000

        with self.assertRaises(ValueError):
            get_sequencer_model(unknown_type, num_classes)
