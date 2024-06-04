from django.test import TestCase, RequestFactory
from django.urls import reverse
from fl_client.views import data_processing, data_processing_faqs, data_processing_models, data_processing_tutorials, deep_learning_faqs, deep_learning_models, deep_learning_tutorials


class IndexViewTests(TestCase):
    def setUp(self):
        self.factory = RequestFactory()

    def test_index_view_uses_correct_template(self):
        """
        Tests if the index view uses the correct template.
        """
        response = self.client.get(reverse('fl_client:index'))
        self.assertTemplateUsed(response, 'core/index.html')

    def test_index_view_context(self):
        """
        Tests if the context of the index view contains the expected elements.
        """
        response = self.client.get(reverse('fl_client:index'))
        self.assertIn('logo', response.context)
        self.assertEqual(response.context['logo'], ["share", "hospital", "data", "cpu", "gpu"])

    def test_data_processing_renders_correct_template(self):
        """
        Tests if the data processing view uses the correct template.
        """
        response = self.client.get(reverse('fl_client:data_processing'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'data_processing/data_processing.html')

    def test_data_processing_context(self):
        """
        Tests if the context of the data processing view contains the expected elements.
        """
        response = self.client.get(reverse('fl_client:data_processing'))
        self.assertEqual(response.status_code, 200)
        self.assertIn('logo', response.context)
        self.assertEqual(response.context['logo'], ["share", "hospital", "data", "cpu", "gpu"])
        self.assertIn('section', response.context)
        self.assertEqual(response.context['section'], "data")

    def test_data_processing_faqs_renders_correct_template(self):
        """
        Tests if the data processing FAQs view uses the correct template.
        """
        response = self.client.get(reverse('fl_client:data_processing_faqs'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'data_processing/data_processing_faqs.html')

    def test_data_processing_faqs_context(self):
        """
        Tests if the context of the data processing FAQs view contains the expected elements.
        """
        response = self.client.get(reverse('fl_client:data_processing_faqs'))
        self.assertEqual(response.status_code, 200)
        self.assertIn('logo', response.context)
        self.assertEqual(response.context['logo'], ["share", "hospital", "data", "cpu", "gpu"])
        self.assertIn('section', response.context)
        self.assertEqual(response.context['section'], "data")

    def test_data_processing_models_renders_correct_template(self):
        """
        Tests if the data processing models view uses the correct template.
        """
        response = self.client.get(reverse('fl_client:data_processing_models'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'data_processing/data_processing_models.html')

    def test_data_processing_models_context(self):
        """
        Tests if the context of the data processing models view contains the expected elements.
        """
        response = self.client.get(reverse('fl_client:data_processing_models'))
        self.assertEqual(response.status_code, 200)
        self.assertIn('logo', response.context)
        self.assertEqual(response.context['logo'], ["share", "hospital", "data", "cpu", "gpu"])
        self.assertIn('section', response.context)
        self.assertEqual(response.context['section'], "data")

    def test_data_processing_tutorials_renders_correct_template(self):
        """
        Tests if the data processing tutorials view uses the correct template.
        """
        response = self.client.get(reverse('fl_client:data_processing_tutorials'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'data_processing/data_processing_tutorials.html')

    def test_data_processing_tutorials_context(self):
        """
        Tests if the context of the data processing tutorials view contains the expected elements.
        """
        response = self.client.get(reverse('fl_client:data_processing_tutorials'))
        self.assertEqual(response.status_code, 200)
        self.assertIn('logo', response.context)
        self.assertEqual(response.context['logo'], ["share", "hospital", "data", "cpu", "gpu"])
        self.assertIn('section', response.context)
        self.assertEqual(response.context['section'], "data")

    def test_deep_learning_faqs_renders_correct_template(self):
        """
        Tests if the deep learning FAQs view uses the correct template.
        """
        response = self.client.get(reverse('fl_client:deep_learning_faqs'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'deep_learning/deep_learning_faqs.html')

    def test_deep_learning_faqs_context(self):
        """
        Tests if the context of the deep learning FAQs view contains the expected elements.
        """
        response = self.client.get(reverse('fl_client:deep_learning_faqs'))
        self.assertEqual(response.status_code, 200)
        self.assertIn('logo', response.context)
        self.assertEqual(response.context['logo'], ["share", "hospital", "data", "cpu", "gpu"])
        self.assertIn('section', response.context)
        self.assertEqual(response.context['section'], "dl")

    def test_deep_learning_models_renders_correct_template(self):
        """
        Tests if the deep learning models view uses the correct template.
        """
        response = self.client.get(reverse('fl_client:deep_learning_models'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'deep_learning/deep_learning_models.html')

    def test_deep_learning_models_context(self):
        """
        Tests if the context of the deep learning models view contains the expected elements.
        """
        response = self.client.get(reverse('fl_client:deep_learning_models'))
        self.assertEqual(response.status_code, 200)
        self.assertIn('logo', response.context)
        self.assertEqual(response.context['logo'], ["share", "hospital", "data", "cpu", "gpu"])
        self.assertIn('section', response.context)
        self.assertEqual(response.context['section'], "dl")

    def test_deep_learning_tutorials_renders_correct_template(self):
        """
        Tests if the deep learning tutorials view uses the correct template.
        """
        response = self.client.get(reverse('fl_client:deep_learning_tutorials'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'deep_learning/deep_learning_tutorials.html')

    def test_deep_learning_tutorials_context(self):
        """
        Tests if the context of the deep learning tutorials view contains the expected elements.
        """
        response = self.client.get(reverse('fl_client:deep_learning_tutorials'))
        self.assertEqual(response.status_code, 200)
        self.assertIn('logo', response.context)
        self.assertEqual(response.context['logo'], ["share", "hospital", "data", "cpu", "gpu"])
        self.assertIn('section', response.context)
        self.assertEqual(response.context['section'], "dl")