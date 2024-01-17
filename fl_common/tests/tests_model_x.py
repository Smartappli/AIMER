import os
import torch
from django.test import TestCase
from fl_common.models.xception import get_xception_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingPartXTestCase(TestCase):
    """Xception Model Unit Tests"""
    def test_all_xception_models(self):
        xception_types = [
            'legacy_xception',
            'xception41',
            'xception65',
            'xception71',
            'xception41p',
            'xception65p'
        ]

        for xception_type in xception_types:
            with self.subTest(xception_type=xception_type):
                model = get_xception_model(xception_type, num_classes=10)
                self.assertTrue(isinstance(model, torch.nn.Module))

    def test_unknown_xception_type(self):
        with self.assertRaises(ValueError):
            get_xception_model('unknown_type', num_classes=10)