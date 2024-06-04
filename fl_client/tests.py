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


class TestIndexView(TestCase):
    def setUp(self):
        self.factory = RequestFactory()

    def test_index_view(self):
        request = self.factory.get(reverse("index"))
        response = index(request)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "core/index.html")

    def test_data_processing_view(self):
        request = self.factory.get(reverse("data_processing"))
        response = data_processing(request)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(
            response, "data_processing/data_processing.html",
        )

    def test_data_processing_faqs_view(self):
        request = self.factory.get(reverse("data_processing_faqs"))
        response = data_processing_faqs(request)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(
            response, "data_processing/data_processing_faqs.html",
        )

    def test_data_processing_models_view(self):
        request = self.factory.get(reverse("data_processing_models"))
        response = data_processing_models(request)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(
            response, "data_processing/data_processing_models.html",
        )

    def test_data_processing_tutorials_view(self):
        request = self.factory.get(reverse("data_processing_tutorials"))
        response = data_processing_tutorials(request)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(
            response, "data_processing/data_processing_tutorials.html",
        )
