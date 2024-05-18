import os
import torch.nn as nn
from django.test import TestCase
from fl_common.models.efficientnet import get_efficientnet_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingEfficientNetTestCase(TestCase):
    """
    Test case class for processing EfficientNet models.
    """

    def test_efficientnet_model(self):
        """
        Test case for validating different configurations of EfficientNet models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        efficientnet_types = [
            "EfficientNetB0",
            "EfficientNetB1",
            "EfficientNetB2",
            "EfficientNetB3",
            "EfficientNetB4",
            "EfficientNetB5",
            "EfficientNetB6",
            "EfficientNetB7",
            "EfficientNetV2S",
            "EfficientNetV2M",
            "EfficientNetV2L",
            "mnasnet_050",
            "mnasnet_075",
            "mnasnet_100",
            "mnasnet_140",
            "semnasnet_050",
            "semnasnet_075",
            "semnasnet_100",
            "semnasnet_140",
            "mnasnet_small",
            "mobilenetv2_035",
            "mobilenetv2_050",
            "mobilenetv2_075",
            "mobilenetv2_100",
            "mobilenetv2_140",
            "mobilenetv2_110d",
            "mobilenetv2_120d",
            "fbnetc_100",
            "spnasnet_100",
            "efficientnet_b0",
            "efficientnet_b1",
            "efficientnet_b2",
            "efficientnet_b3",
            "efficientnet_b4",
            "efficientnet_b5",
            "efficientnet_b6",
            "efficientnet_b7",
            "efficientnet_b8",
            "efficientnet_l2",
            "efficientnet_b0_gn",
            "efficientnet_b0_g8_gn",
            "efficientnet_b0_g16_evos",
            "efficientnet_b3_gn",
            "efficientnet_b3_g8_gn",
            "efficientnet_es",
            "efficientnet_es_pruned",
            "efficientnet_em",
            "efficientnet_el",
            "efficientnet_el_pruned",
            "efficientnet_cc_b0_4e",
            "efficientnet_cc_b0_8e",
            "efficientnet_cc_b1_8e",
            "efficientnet_lite0",
            "efficientnet_lite1",
            "efficientnet_lite2",
            "efficientnet_lite3",
            "efficientnet_lite4",
            "efficientnet_b1_pruned",
            "efficientnet_b2_pruned",
            "efficientnet_b3_pruned",
            "efficientnetv2_rw_t",
            "gc_efficientnetv2_rw_t",
            "efficientnetv2_rw_s",
            "efficientnetv2_rw_m",
            "efficientnetv2_s",
            "efficientnetv2_m",
            "efficientnetv2_l",
            "efficientnetv2_xl",
            "tf_efficientnet_b0",
            "tf_efficientnet_b1",
            "tf_efficientnet_b2",
            "tf_efficientnet_b3",
            "tf_efficientnet_b4",
            "tf_efficientnet_b5",
            "tf_efficientnet_b6",
            "tf_efficientnet_b7",
            "tf_efficientnet_b8",
            "tf_efficientnet_l2",
            "tf_efficientnet_es",
            "tf_efficientnet_em",
            "tf_efficientnet_el",
            "tf_efficientnet_cc_b0_4e",
            "tf_efficientnet_cc_b0_8e",
            "tf_efficientnet_cc_b1_8e",
            "tf_efficientnet_lite0",
            "tf_efficientnet_lite1",
            "tf_efficientnet_lite2",
            "tf_efficientnet_lite3",
            "tf_efficientnet_lite4",
            "tf_efficientnetv2_s",
            "tf_efficientnetv2_m",
            "tf_efficientnetv2_l",
            "tf_efficientnetv2_xl",
            "tf_efficientnetv2_b0",
            "tf_efficientnetv2_b1",
            "tf_efficientnetv2_b2",
            "tf_efficientnetv2_b3",
            "mixnet_s",
            "mixnet_m",
            "mixnet_l",
            "mixnet_xl",
            "mixnet_xxl",
            "tf_mixnet_s",
            "tf_mixnet_m",
            "tf_mixnet_l",
            "tinynet_a",
            "tinynet_b",
            "tinynet_c",
            "tinynet_d",
            "tinynet_e",
        ]
        num_classes = 10  # You can adjust the number of classes as needed

        for efficientnet_type in efficientnet_types:
            with self.subTest(efficientnet_type=efficientnet_type):
                model = get_efficientnet_model(efficientnet_type, num_classes)
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_efficientnet_unknown_architecture(self):
        """
        Test case for handling unknown EfficientNet architectures in get_efficientnet_model function.

        Raises:
            AssertionError: If any of the assertions fail.
            ValueError: If an unknown EfficientNet architecture is provided.
        """
        efficientnet_type = "UnknownArchitecture"
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            get_efficientnet_model(efficientnet_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f"Unknown EfficientNet Architecture: {efficientnet_type}",
        )
