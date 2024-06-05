from django.test import Client, TestCase
from django.urls import reverse


class TestUrls(TestCase):
    """
    Test case for the URLs of the 'fl_client' Django app.
    """

    def setUp(self):
        """
        Set up the test case with a Django test client.
        """
        self.client = Client()

    def test_index_url(self):
        """
        Test the index URL.
        """
        response = self.client.get(reverse("fl_client:index"))
        self.assertEqual(response.status_code, 200)

    def test_data_processing_url(self):
        """
        Test the data processing URL.
        """
        response = self.client.get(reverse("fl_client:data_processing"))
        self.assertEqual(response.status_code, 200)

    def test_data_processing_faqs_url(self):
        """
        Test the data processing FAQs URL.
        """
        response = self.client.get(reverse("fl_client:data_processing_faqs"))
        self.assertEqual(response.status_code, 200)

    def test_data_processing_models_url(self):
        """
        Test the data processing models URL.
        """
        response = self.client.get(reverse("fl_client:data_processing_models"))
        self.assertEqual(response.status_code, 200)

    def test_data_processing_tutorials_url(self):
        """
        Test the data processing tutorials URL.
        """
        response = self.client.get(
            reverse("fl_client:data_processing_tutorials"),
        )
        self.assertEqual(response.status_code, 200)

    def test_deep_learning_url(self):
        """
        Test the deep learning URL.
        """
        response = self.client.get(reverse("fl_client:deep_learning"))
        self.assertEqual(response.status_code, 200)

    def test_deep_learning_faqs_url(self):
        """
        Test the deep learning FAQs URL.
        """
        response = self.client.get(reverse("fl_client:deep_learning_faqs"))
        self.assertEqual(response.status_code, 200)

    def test_deep_learning_models_url(self):
        """
        Test the deep learning models URL.
        """
        response = self.client.get(reverse("fl_client:deep_learning_models"))
        self.assertEqual(response.status_code, 200)

    def test_deep_learning_tutorials_url(self):
        """
        Test the deep learning tutorials URL.
        """
        response = self.client.get(reverse("fl_client:deep_learning_tutorials"))
        self.assertEqual(response.status_code, 200)

    def test_machine_learning_url(self):
        """
        Test the machine learning URL.
        """
        response = self.client.get(reverse("fl_client:machine_learning"))
        self.assertEqual(response.status_code, 200)

    def test_machine_learning_faqs_url(self):
        """
        Test the machine learning FAQs URL.
        """
        response = self.client.get(reverse("fl_client:machine_learning_faqs"))
        self.assertEqual(response.status_code, 200)

    def test_machine_learning_models_url(self):
        """
        Test the machine learning models URL.
        """
        response = self.client.get(reverse("fl_client:machine_learning_models"))
        self.assertEqual(response.status_code, 200)

    def test_machine_learning_tutorials_url(self):
        """
        Test the machine learning tutorials URL.
        """
        response = self.client.get(
            reverse("fl_client:machine_learning_tutorials"),
        )
        self.assertEqual(response.status_code, 200)

    def test_natural_language_processing_url(self):
        """
        Test the natural language processing URL.
        """
        response = self.client.get(
            reverse("fl_client:natural_language_processing"),
        )
        self.assertEqual(response.status_code, 200)

    def test_natural_language_processing_faqs_url(self):
        """
        Test the natural language processing FAQs URL.
        """
        response = self.client.get(
            reverse("fl_client:natural_language_processing_faqs"),
        )
        self.assertEqual(response.status_code, 200)

    def test_natural_language_processing_models_url(self):
        """
        Test the natural language processing models URL.
        """
        response = self.client.get(
            reverse("fl_client:natural_language_processing_models"),
        )
        self.assertEqual(response.status_code, 200)

    def test_natural_language_processing_tutorials_url(self):
        """
        Test the natural language processing tutorials URL.
        """
        response = self.client.get(
            reverse("fl_client:natural_language_processing_tutorials"),
        )
        self.assertEqual(response.status_code, 200)

    def test_import_url(self):
        """
        Test the import URL.
        """
        response = self.client.get(reverse("fl_client:import"))
        self.assertEqual(response.status_code, 200)

    def test_download_url(self):
        """
        Test the download URL.
        """
        response = self.client.get(reverse("fl_client:download"))
        self.assertEqual(response.status_code, 200)
