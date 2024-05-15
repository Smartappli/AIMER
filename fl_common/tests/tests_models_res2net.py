import os
from django.test import TestCase
from fl_common.models.res2net import get_res2net_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingRes2netTestCase(TestCase):
    """
    Test case class for processing Res2Net models.
    """

    def test_known_res2net_type(self):
        """
        Test if the function returns a valid Res2Net model for known Res2Net types.
        """
        res2net_types = [
            'res2net50_26w_4s',
            'res2net101_26w_4s',
            'res2net50_26w_6s',
            'res2net50_26w_8s',
            'res2net50_48w_2s',
            'res2net50_14w_8s',
            'res2next50',
            'res2net50d',
            'res2net101d']
        num_classes = 10  # Just for testing purposes

        for res2net_type in res2net_types:
            res2net_model = get_res2net_model(res2net_type, num_classes)
            self.assertIsNotNone(res2net_model)
            # Check if the model has the expected number of output classes
            self.assertEqual(res2net_model.num_classes, num_classes)

    def test_unknown_res2net_type(self):
        """
        Test if the function raises a ValueError for an unknown Res2Net type.
        """
        unknown_res2net_type = 'unknown_res2net_type'
        num_classes = 10

        with self.assertRaises(ValueError):
            get_res2net_model(unknown_res2net_type, num_classes)
