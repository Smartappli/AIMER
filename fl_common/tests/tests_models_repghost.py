import os
from django.test import TestCase
from fl_common.models.repghost import get_repghost_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingRepghostTestCase(TestCase):
    # Repghost Models Unit tests
    def test_known_repghost_type(self):
        repghost_types = ['repghostnet_050', 'repghostnet_058', 'repghostnet_080',
                          'repghostnet_100', 'repghostnet_111', 'repghostnet_130',
                          'repghostnet_150', 'repghostnet_200']
        num_classes = 10  # Just for testing purposes

        for repghost_type in repghost_types:
            repghost_model = get_repghost_model(repghost_type, num_classes)
            self.assertIsNotNone(repghost_model)
            # Check if the model has the expected number of output classes
            self.assertEqual(repghost_model.num_classes, num_classes)

    def test_unknown_repghost_type(self):
        unknown_repghost_type = 'unknown_repghost_type'
        num_classes = 10

        with self.assertRaises(ValueError):
            get_repghost_model(unknown_repghost_type, num_classes)

