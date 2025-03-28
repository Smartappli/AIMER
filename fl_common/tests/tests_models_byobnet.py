import os

from django.test import TestCase

from fl_common.models.byobnet import get_byobnet_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingByobnetTestCase(TestCase):
    """
    Test case class for processing Byobnet models.
    """

    def test_known_byobnet_types(self):
        """
        Vérifie si les modèles sont correctement créés pour les types de Byobnet connus.
        """
        known_types = [
            "gernet_l",
            "gernet_m",
            "gernet_s",
            "repvgg_a0",
            "repvgg_a1",
            "repvgg_a2",
            "repvgg_b0",
            "repvgg_b1",
            "repvgg_b1g4",
            "repvgg_b2",
            "repvgg_b2g4",
            "repvgg_b3",
            "repvgg_b3g4",
            "repvgg_d2se",
            "resnet51q",
            "resnet61q",
            "resnext26ts",
            "gcresnext26ts",
            "seresnext26ts",
            "eca_resnext26ts",
            "bat_resnext26ts",
            "resnet32ts",
            "resnet33ts",
            "gcresnet33ts",
            "seresnet33ts",
            "eca_resnet33ts",
            "gcresnet50t",
            "gcresnext50ts",
            "regnetz_b16",
            "regnetz_c16",
            "regnetz_d32",
            "regnetz_d8",
            "regnetz_e8",
            "regnetz_b16_evos",
            "regnetz_c16_evos",
            "regnetz_d8_evos",
            "mobileone_s0",
            "mobileone_s1",
            "mobileone_s2",
            "mobileone_s3",
            "mobileone_s4",
            "resnet50_clip",
            "resnet101_clip",
            "resnet50x4_clip",
            "resnet50x16_clip",
            "resnet50x64_clip",
            "resnet50_clip_gap",
            "resnet101_clip_gap",
            "resnet50x4_clip_gap",
            "resnet50x16_clip_gap",
            "resnet50x64_clip_gap",
            "resnet50_mlp",
            "test_byobnet",
        ]
        num_classes = 1000
        for byobnet_type in known_types:
            with self.subTest(byobnet_type=byobnet_type):
                model = get_byobnet_model(byobnet_type, num_classes)
                self.assertIsNotNone(model)
                # Vérifier que le modèle retourné a bien le nombre de classes
                # spécifié
                self.assertEqual(model.num_classes, num_classes)

    def test_unknown_byobnet_type(self):
        """
        Vérifie si la fonction lève une exception ValueError pour un type de Byobnet inconnu.
        """
        unknown_type = "unknown_type"
        num_classes = 1000
        with self.assertRaises(ValueError):
            get_byobnet_model(unknown_type, num_classes)
