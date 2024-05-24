import os
from django.test import TestCase
from fl_common.models.densenet import get_densenet_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingDenseNetTestCase(TestCase):
    """
    Test case class for processing Densenet models.
    """

    def test_densenet_model(self):
        """
        Test case for creating DenseNet models.

        Iterates through different DenseNet architectures and checks if the created model is an instance
        of `nn.Module`.

        Raises:
            AssertionError: If the assertion fails.
        """
        # List of DenseNet architectures to test
        densenet_types = [
            "DenseNet121",
            "DenseNet161",
            "DenseNet169",
            "DenseNet201",
            "densenet121",
            "densenetblur121d",
            "densenet169",
            "densenet201",
            "densenet161",
            "densenet264d",
        ]
        num_classes = 10  # You can adjust the number of classes as needed

        for densenet_type in densenet_types:
            with self.subTest(densenet_type=densenet_type):
                model = get_densenet_model(densenet_type, num_classes)
                try:
                    model = get_densenet_model(densenet_type, num_classes)
                    self.assertIsNotNone(model)
                except ValueError:
                    self.fail(
                        f"{densenet_type} should be a known Densenet architecture."
                    )

    def test_densenet_unknown_architecture(self):
        """
        Test case for handling unknown DenseNet architecture.

        Raises:
            ValueError: If an unknown DenseNet architecture is encountered.
            AssertionError: If the assertion fails.
        """
        densenet_type = "UnknownArchitecture"
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            get_densenet_model(densenet_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f"Unknown DenseNet Architecture : {densenet_type}",
        )
