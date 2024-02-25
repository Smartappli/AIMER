import os
import torch.nn as nn
from django.test import TestCase
from fl_common.models.byoanet import get_byoanet_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingByaonetTestCase(TestCase):
    """
    Test case class for processing Byoanet models.
    """

    def test_known_byoanet_types(self):
        """
        Vérifie si les modèles sont correctement créés pour les types de Byoanet connus.
        """
        known_types = [
            'botnet26t_256',
            'sebotnet33ts_256',
            'botnet50ts_256',
            'eca_botnext26ts_256',
            'halonet_h1',
            'halonet26t',
            'sehalonet33ts',
            'halonet50ts',
            'eca_halonext26ts',
            'lambda_resnet26t',
            'lambda_resnet50ts',
            'lambda_resnet26rpt_256',
            'haloregnetz_b',
            'lamhalobotnet50ts_256',
            'halo2botnet50ts_256'
        ]
        num_classes = 1000
        for byoanet_type in known_types:
            with self.subTest(byoanet_type=byoanet_type):
                model = get_byoanet_model(byoanet_type, num_classes)
                self.assertIsNotNone(model)
                # Vérifier que le modèle retourné a bien le nombre de classes spécifié
                self.assertEqual(model.num_classes, num_classes)

    def test_unknown_byoanet_type(self):
        """
        Vérifie si la fonction lève une exception ValueError pour un type de Byoanet inconnu.
        """
        unknown_type = 'unknown_type'
        num_classes = 1000
        with self.assertRaises(ValueError):
            get_byoanet_model(unknown_type, num_classes)
