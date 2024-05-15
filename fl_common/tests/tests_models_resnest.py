import os
from django.test import TestCase
from fl_common.models.resnest import get_resnest_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingResnestTestCase(TestCase):
    """
    Test case class for processing Resnest models.
    """

    def test_known_res2net_type(self):
        """
        Test if the function returns a valid Res2Net model for known Res2Net types.
        """
        resnest_types = [
            'resnest14d',
            'resnest26d',
            'resnest50d',
            'resnest101e',
            'resnest200e',
            'resnest269e',
            'resnest50d_4s2x40d',
            'resnest50d_1s4x24d']
        num_classes = 10  # Just for testing purposes

        for resnest_type in resnest_types:
            resnest_model = get_resnest_model(resnest_type, num_classes)
            self.assertIsNotNone(resnest_model)
            # Check if the model has the expected number of output classes
            self.assertEqual(resnest_model.num_classes, num_classes)

    def test_unknown_resnest_type(self):
        """
        Test if the function raises a ValueError for an unknown Resnest type.
        """
        unknown_resnest_type = 'unknown_resnest_type'
        num_classes = 10

        with self.assertRaises(ValueError):
            get_resnest_model(unknown_resnest_type, num_classes)
