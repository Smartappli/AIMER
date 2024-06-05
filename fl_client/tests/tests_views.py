from django.test import RequestFactory, TestCase
from django.urls import reverse

from fl_client.forms import NLPEmotionalAnalysisForm, NLPTextGenerationForm


class IndexViewTests(TestCase):
    def setUp(self):
        self.factory = RequestFactory()

    def test_index_view_uses_correct_template(self):
        """
        Tests if the index view uses the correct template.
        """
        response = self.client.get(reverse("fl_client:index"))
        self.assertTemplateUsed(response, "core/index.html")

    def test_index_view_context(self):
        """
        Tests if the context of the index view contains the expected elements.
        """
        response = self.client.get(reverse("fl_client:index"))
        self.assertIn("logo", response.context)
        self.assertEqual(
            response.context["logo"],
            ["share", "hospital", "data", "cpu", "gpu"],
        )

    def test_data_processing_renders_correct_template(self):
        """
        Tests if the data processing view uses the correct template.
        """
        response = self.client.get(reverse("fl_client:data_processing"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(
            response,
            "data_processing/data_processing.html",
        )

    def test_data_processing_context(self):
        """
        Tests if the context of the data processing view contains the expected elements.
        """
        response = self.client.get(reverse("fl_client:data_processing"))
        self.assertEqual(response.status_code, 200)
        self.assertIn("logo", response.context)
        self.assertEqual(
            response.context["logo"],
            ["share", "hospital", "data", "cpu", "gpu"],
        )
        self.assertIn("section", response.context)
        self.assertEqual(response.context["section"], "data")

    def test_data_processing_faqs_renders_correct_template(self):
        """
        Tests if the data processing FAQs view uses the correct template.
        """
        response = self.client.get(reverse("fl_client:data_processing_faqs"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(
            response,
            "data_processing/data_processing_faqs.html",
        )

    def test_data_processing_faqs_context(self):
        """
        Tests if the context of the data processing FAQs view contains the expected elements.
        """
        response = self.client.get(reverse("fl_client:data_processing_faqs"))
        self.assertEqual(response.status_code, 200)
        self.assertIn("logo", response.context)
        self.assertEqual(
            response.context["logo"],
            ["share", "hospital", "data", "cpu", "gpu"],
        )
        self.assertIn("section", response.context)
        self.assertEqual(response.context["section"], "data")

    def test_data_processing_models_renders_correct_template(self):
        """
        Tests if the data processing models view uses the correct template.
        """
        response = self.client.get(reverse("fl_client:data_processing_models"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(
            response,
            "data_processing/data_processing_models.html",
        )

    def test_data_processing_models_context(self):
        """
        Tests if the context of the data processing models view contains the expected elements.
        """
        response = self.client.get(reverse("fl_client:data_processing_models"))
        self.assertEqual(response.status_code, 200)
        self.assertIn("logo", response.context)
        self.assertEqual(
            response.context["logo"],
            ["share", "hospital", "data", "cpu", "gpu"],
        )
        self.assertIn("section", response.context)
        self.assertEqual(response.context["section"], "data")

    def test_data_processing_tutorials_renders_correct_template(self):
        """
        Tests if the data processing tutorials view uses the correct template.
        """
        response = self.client.get(
            reverse("fl_client:data_processing_tutorials"),
        )
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(
            response,
            "data_processing/data_processing_tutorials.html",
        )

    def test_data_processing_tutorials_context(self):
        """
        Tests if the context of the data processing tutorials view contains the expected elements.
        """
        response = self.client.get(
            reverse("fl_client:data_processing_tutorials"),
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("logo", response.context)
        self.assertEqual(
            response.context["logo"],
            ["share", "hospital", "data", "cpu", "gpu"],
        )
        self.assertIn("section", response.context)
        self.assertEqual(response.context["section"], "data")

    def test_deep_learning_faqs_renders_correct_template(self):
        """
        Tests if the deep learning FAQs view uses the correct template.
        """
        response = self.client.get(reverse("fl_client:deep_learning_faqs"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(
            response,
            "deep_learning/deep_learning_faqs.html",
        )

    def test_deep_learning_faqs_context(self):
        """
        Tests if the context of the deep learning FAQs view contains the expected elements.
        """
        response = self.client.get(reverse("fl_client:deep_learning_faqs"))
        self.assertEqual(response.status_code, 200)
        self.assertIn("logo", response.context)
        self.assertEqual(
            response.context["logo"],
            ["share", "hospital", "data", "cpu", "gpu"],
        )
        self.assertIn("section", response.context)
        self.assertEqual(response.context["section"], "dl")

    def test_deep_learning_models_renders_correct_template(self):
        """
        Tests if the deep learning models view uses the correct template.
        """
        response = self.client.get(reverse("fl_client:deep_learning_models"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(
            response,
            "deep_learning/deep_learning_models.html",
        )

    def test_deep_learning_models_context(self):
        """
        Tests if the context of the deep learning models view contains the expected elements.
        """
        response = self.client.get(reverse("fl_client:deep_learning_models"))
        self.assertEqual(response.status_code, 200)
        self.assertIn("logo", response.context)
        self.assertEqual(
            response.context["logo"],
            ["share", "hospital", "data", "cpu", "gpu"],
        )
        self.assertIn("section", response.context)
        self.assertEqual(response.context["section"], "dl")

    def test_deep_learning_tutorials_renders_correct_template(self):
        """
        Tests if the deep learning tutorials view uses the correct template.
        """
        response = self.client.get(reverse("fl_client:deep_learning_tutorials"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(
            response,
            "deep_learning/deep_learning_tutorials.html",
        )

    def test_deep_learning_tutorials_context(self):
        """
        Tests if the context of the deep learning tutorials view contains the expected elements.
        """
        response = self.client.get(reverse("fl_client:deep_learning_tutorials"))
        self.assertEqual(response.status_code, 200)
        self.assertIn("logo", response.context)
        self.assertEqual(
            response.context["logo"],
            ["share", "hospital", "data", "cpu", "gpu"],
        )
        self.assertIn("section", response.context)
        self.assertEqual(response.context["section"], "dl")

    def test_machine_learning_faqs_renders_correct_template(self):
        """
        Tests if the machine learning FAQs view uses the correct template.
        """
        response = self.client.get(reverse("fl_client:machine_learning_faqs"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(
            response,
            "machine_learning/machine_learning_faqs.html",
        )

    def test_machine_learning_faqs_context(self):
        """
        Tests if the context of the machine learning FAQs view contains the expected elements.
        """
        response = self.client.get(reverse("fl_client:machine_learning_faqs"))
        self.assertEqual(response.status_code, 200)
        self.assertIn("logo", response.context)
        self.assertEqual(
            response.context["logo"],
            ["share", "hospital", "data", "cpu", "gpu"],
        )
        self.assertIn("section", response.context)
        self.assertEqual(response.context["section"], "ml")

    def test_machine_learning_models_renders_correct_template(self):
        """
        Tests if the machine learning models view uses the correct template.
        """
        response = self.client.get(reverse("fl_client:machine_learning_models"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(
            response,
            "machine_learning/machine_learning_models.html",
        )

    def test_machine_learning_models_context(self):
        """
        Tests if the context of the machine learning models view contains the expected elements.
        """
        response = self.client.get(reverse("fl_client:machine_learning_models"))
        self.assertEqual(response.status_code, 200)
        self.assertIn("logo", response.context)
        self.assertEqual(
            response.context["logo"],
            ["share", "hospital", "data", "cpu", "gpu"],
        )
        self.assertIn("section", response.context)
        self.assertEqual(response.context["section"], "ml")

    def test_machine_learning_tutorials_renders_correct_template(self):
        """
        Tests if the machine learning tutorials view uses the correct template.
        """
        response = self.client.get(
            reverse("fl_client:machine_learning_tutorials"),
        )
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(
            response,
            "machine_learning/machine_learning_tutorials.html",
        )

    def test_machine_learning_tutorials_context(self):
        """
        Tests if the context of the machine learning tutorials view contains the expected elements.
        """
        response = self.client.get(
            reverse("fl_client:machine_learning_tutorials"),
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("logo", response.context)
        self.assertEqual(
            response.context["logo"],
            ["share", "hospital", "data", "cpu", "gpu"],
        )
        self.assertIn("section", response.context)
        self.assertEqual(response.context["section"], "ml")

    def test_natural_language_processing_renders_correct_template(self):
        """
        Tests if the natural language processing view uses the correct template.
        """
        response = self.client.get(
            reverse("fl_client:natural_language_processing"),
        )
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(
            response,
            "natural_language_processing/natural_language_processing.html",
        )

    def test_natural_language_processing_context(self):
        """
        Tests if the context of the natural language processing view contains the expected elements.
        """
        response = self.client.get(
            reverse("fl_client:natural_language_processing"),
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("logo", response.context)
        self.assertEqual(
            response.context["logo"],
            ["share", "hospital", "data", "cpu", "gpu"],
        )
        self.assertIn("section", response.context)
        self.assertEqual(response.context["section"], "nlp")
        self.assertIn("form1", response.context)
        self.assertIsInstance(response.context["form1"], NLPTextGenerationForm)
        self.assertIn("form2", response.context)
        self.assertIsInstance(
            response.context["form2"],
            NLPEmotionalAnalysisForm,
        )
        self.assertIn("pdf", response.context)
        self.assertTrue(response.context["pdf"])

    def test_natural_language_processing_faqs_renders_correct_template(self):
        """
        Tests if the natural language processing FAQs view uses the correct template.
        """
        response = self.client.get(
            reverse("fl_client:natural_language_processing_faqs"),
        )
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(
            response,
            "natural_language_processing/natural_language_processing_faqs.html",
        )

    def test_natural_language_processing_faqs_context(self):
        """
        Tests if the context of the natural language processing FAQs view contains the expected elements.
        """
        response = self.client.get(
            reverse("fl_client:natural_language_processing_faqs"),
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("logo", response.context)
        self.assertEqual(
            response.context["logo"],
            ["share", "hospital", "data", "cpu", "gpu"],
        )
        self.assertIn("section", response.context)
        self.assertEqual(response.context["section"], "nlp")

    def test_natural_language_processing_models_renders_correct_template(self):
        """
        Tests if the natural language processing models view uses the correct template.
        """
        response = self.client.get(
            reverse("fl_client:natural_language_processing_models"),
        )
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(
            response,
            "natural_language_processing/natural_language_processing_models.html",
        )

    def test_natural_language_processing_models_context(self):
        """
        Tests if the context of the natural language processing models view contains the expected elements.
        """
        response = self.client.get(
            reverse("fl_client:natural_language_processing_models"),
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("logo", response.context)
        self.assertEqual(
            response.context["logo"],
            ["share", "hospital", "data", "cpu", "gpu"],
        )
        self.assertIn("section", response.context)
        self.assertEqual(response.context["section"], "nlp")

    def test_natural_language_processing_tutorials_renders_correct_template(
        self,
    ):
        """
        Tests if the natural language processing tutorials view uses the correct template.
        """
        response = self.client.get(
            reverse("fl_client:natural_language_processing_tutorials"),
        )
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(
            response,
            "natural_language_processing/natural_language_processing_tutorials.html",
        )

    def test_natural_language_processing_tutorials_context(self):
        """
        Tests if the context of the natural language processing tutorials view contains the expected elements.
        """
        response = self.client.get(
            reverse("fl_client:natural_language_processing_tutorials"),
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("logo", response.context)
        self.assertEqual(
            response.context["logo"],
            ["share", "hospital", "data", "cpu", "gpu"],
        )
        self.assertIn("section", response.context)
        self.assertEqual(response.context["section"], "nlp")
