import os
from django.test import TestCase
from torch import nn
from unittest.mock import patch

# Assurez-vous d'importer la fonction get_mambaout_model depuis le bon module
from your_module import get_mambaout_model

# Désactivation de l'avertissement pour les liens symboliques
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class MambaoutModelTestCase(TestCase):
    """
    Test case class for processing Mambaout models.
    """

    def test_valid_architecture(self):
        """Test for valid Mambaout architectures."""
        valid_architectures = [
            "mambaout_femto",
            "mambaout_kobe",
            "mambaout_tiny",
            "mambaout_small",
            "mambaout_base",
            "mambaout_small_rw",
            "mambaout_base_short_rw",
            "mambaout_base_tall_rw",
            "mambaout_base_wide_rw",
            "mambaout_base_plus_rw",
            "test_mambaout",
        ]
        num_classes = 1000  # Remplacez par le nombre de classes approprié

        for mambaout_type in valid_architectures:
            with self.subTest(mambaout_type=mambaout_type):
                result = get_mambaout_model(mambaout_type, num_classes)
                self.assertIsInstance(result, nn.Module)

    def test_unknown_architecture(self):
        """Test for an unknown Mambaout architecture."""
        mambaout_type = "unknown_architecture"
        num_classes = 1000  # Remplacez par le nombre de classes approprié
        with self.assertRaises(ValueError):
            get_mambaout_model(mambaout_type, num_classes)

    def test_valid_architecture_custom_classes(self):
        """Test for a valid Mambaout architecture with a custom number of classes."""
        valid_architectures = [
            "mambaout_femto",
            "mambaout_kobe",
            "mambaout_tiny",
            "mambaout_small",
            "mambaout_base",
            "mambaout_small_rw",
            "mambaout_base_short_rw",
            "mambaout_base_tall_rw",
            "mambaout_base_wide_rw",
            "mambaout_base_plus_rw",
            "test_mambaout",
        ]
        num_classes = 500  # Remplacez par un nombre personnalisé de classes

        for mambaout_type in valid_architectures:
            with self.subTest(mambaout_type=mambaout_type):
                result = get_mambaout_model(mambaout_type, num_classes)
                self.assertIsInstance(result, nn.Module)

    @patch("timm.create_model")
    def test_runtime_error_loading_pretrained_model(self, mock_create_model):
        """Test that if loading the pretrained model fails, the model is created without pretrained weights."""
        mambaout_type = "mambaout_base"
        num_classes = 5

        # Simule un RuntimeError lors du chargement du modèle pré-entraîné
        mock_create_model.side_effect = RuntimeError(
            "Pretrained model not found"
        )

        model = get_mambaout_model(mambaout_type, num_classes)

        # Vérifie que la création du modèle a bien été appelée avec pretrained=False
        mock_create_model.assert_called_once_with(
            mambaout_type, pretrained=False, num_classes=num_classes
        )
        self.assertEqual(model, mock_create_model.return_value)
