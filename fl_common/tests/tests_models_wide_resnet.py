import os

from django.test import TestCase

from fl_common.models.wide_resnet import get_wide_resnet_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingWideResnetTestCase(TestCase):
    """
    Test case class for processing Wide Resnet models.
    """

    def test_get_wide_resnet_model(self):
        """
        Test case for obtaining a Wide ResNet model.

        Raises:
            AssertionError: If the assertion fails.
        """
        wide_resnet_model = get_wide_resnet_model("Wide_ResNet50_2", 1000)
        self.assertIsNotNone(wide_resnet_model, msg="Wide ResNet KO")
