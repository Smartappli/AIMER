from django.test import TestCase, Client
from django.urls import reverse


class TestUrls(TestCase):
    def setUp(self):
        self.client = Client()

    def test_index_url(self):
        response = self.client.get(reverse("fl_client:index"))
        self.assertEqual(response.status_code, 200)

    def test_data_processing_url(self):
        response = self.client.get(reverse("fl_client:data_processing"))
        self.assertEqual(response.status_code, 200)

    def test_data_processing_faqs_url(self):
        response = self.client.get(reverse("fl_client:data_processing_faqs"))
        self.assertEqual(response.status_code, 200)

    def test_data_processing_models_url(self):
        response = self.client.get(reverse("fl_client:data_processing_models"))
        self.assertEqual(response.status_code, 200)

    def test_data_processing_tutorials_url(self):
        response = self.client.get(
            reverse("fl_client:data_processing_tutorials"),
        )
        self.assertEqual(response.status_code, 200)

    def test_deep_learning_url(self):
        response = self.client.get(reverse("fl_client:deep_learning"))
        self.assertEqual(response.status_code, 200)

    def test_deep_learning_faqs_url(self):
        response = self.client.get(reverse("fl_client:deep_learning_faqs"))
        self.assertEqual(response.status_code, 200)

    def test_deep_learning_models_url(self):
        response = self.client.get(reverse("fl_client:deep_learning_models"))
        self.assertEqual(response.status_code, 200)

    def test_deep_learning_tutorials_url(self):
        response = self.client.get(reverse("fl_client:deep_learning_tutorials"))
        self.assertEqual(response.status_code, 200)

    def test_machine_learning_url(self):
        response = self.client.get(reverse("fl_client:machine_learning"))
        self.assertEqual(response.status_code, 200)

    def test_machine_learning_faqs_url(self):
        response = self.client.get(reverse("fl_client:machine_learning_faqs"))
        self.assertEqual(response.status_code, 200)

    def test_machine_learning_models_url(self):
        response = self.client.get(reverse("fl_client:machine_learning_models"))
        self.assertEqual(response.status_code, 200)

    def test_machine_learning_tutorials_url(self):
        response = self.client.get(
            reverse("fl_client:machine_learning_tutorials"),
        )
        self.assertEqual(response.status_code, 200)

    def test_natural_language_processing_url(self):
        response = self.client.get(
            reverse("fl_client:natural_language_processing"),
        )
        self.assertEqual(response.status_code, 200)

    def test_natural_language_processing_faqs_url(self):
        response = self.client.get(
            reverse("fl_client:natural_language_processing_faqs"),
        )
        self.assertEqual(response.status_code, 200)

    def test_natural_language_processing_models_url(self):
        response = self.client.get(
            reverse("fl_client:natural_language_processing_models"),
        )
        self.assertEqual(response.status_code, 200)

    def test_natural_language_processing_tutorials_url(self):
        response = self.client.get(
            reverse("fl_client:natural_language_processing_tutorials"),
        )
        self.assertEqual(response.status_code, 200)

    def test_register_url(self):
        response = self.client.get(reverse("fl_client:register"))
        self.assertEqual(response.status_code, 200)

    # def test_edit_url(self):
    #    response = self.client.get(reverse("fl_client:edit"))
    #    self.assertEqual(response.status_code, 200)

    def test_import_url(self):
        response = self.client.get(reverse("fl_client:import"))
        self.assertEqual(response.status_code, 200)

    def test_download_url(self):
        response = self.client.get(reverse("fl_client:download"))
        self.assertEqual(response.status_code, 200)
