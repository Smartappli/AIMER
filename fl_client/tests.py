from django.test import RequestFactory, TestCase
from django.urls import reverse, resolve
from .views import (
    index,
    data_processing,
    data_processing_faqs,
    data_processing_models,
    data_processing_tutorials,
)
from fl_client import views


class TestUrls(TestCase):
    def test_index_url(self):
        path = reverse("fl_client:index")
        self.assertEqual(resolve(path).func, views.index)

    def test_data_processing_url(self):
        path = reverse("fl_client:data_processing")
        self.assertEqual(resolve(path).func, views.data_processing)

    def test_data_processing_faqs_url(self):
        path = reverse("fl_client:data_processing_faqs")
        self.assertEqual(resolve(path).func, views.data_processing_faqs)

    def test_data_processing_models_url(self):
        path = reverse("fl_client:data_processing_models")
        self.assertEqual(resolve(path).func, views.data_processing_models)

    def test_data_processing_tutorials_url(self):
        path = reverse("fl_client:data_processing_tutorials")
        self.assertEqual(resolve(path).func, views.data_processing_tutorials)


