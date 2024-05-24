import os
from django.test import TestCase
from fl_common.models.mobilenet import get_mobilenet_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingMobilenetTestCase(TestCase):
    """
    Test case class for processing Mobilenet models.
    """

    def test_get_mobilenet_model(self):
        """
        Test case for obtaining various MobileNet models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # List of MobileNet model types to test
        mobilenet_types = [
            "MobileNet_V2",
            "MobileNet_V3_Small",
            "MobileNet_V3_Large",
            "mobilenetv3_large_075",
            "mobilenetv3_large_100",
            "mobilenetv3_small_050",
            "mobilenetv3_small_075",
            "mobilenetv3_small_100",
            "mobilenetv3_rw",
            "tf_mobilenetv3_large_075",
            "tf_mobilenetv3_large_100",
            "tf_mobilenetv3_large_minimal_100",
            "tf_mobilenetv3_small_075",
            "tf_mobilenetv3_small_100",
            "tf_mobilenetv3_small_minimal_100",
            "fbnetv3_b",
            "fbnetv3_d",
            "fbnetv3_g",
            "lcnet_035",
            "lcnet_050",
            "lcnet_075",
            "lcnet_100",
            "lcnet_150",
        ]
        num_classes = 10  # You can adjust the number of classes as needed

        # Loop through each MobileNet model type
        for mobilenet_type in mobilenet_types:
            with self.subTest(mobilenet_type=mobilenet_type):
                try:
                    model = get_mobilenet_model(mobilenet_type, num_classes)
                    self.assertIsNotNone(model)
                except ValueError:
                    self.fail(
                        f"{mobilenet_type} should be a known MobileNet architecture."
                    )

    def test_mobilenet_unknown_architecture(self):
        """
        Test case for handling unknown MobileNet architecture in get_mobilenet_model function.

        Raises:
            AssertionError: If the assertion fails.
            ValueError: If an unknown MobileNet architecture is provided.
        """
        mobilenet_type = "UnknownArchitecture"
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            # Attempt to get a MobileNet model with an unknown architecture
            get_mobilenet_model(mobilenet_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f"Unknown MobileNet Architecture : {mobilenet_type}",
        )
