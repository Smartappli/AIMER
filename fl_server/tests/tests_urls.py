from django.test import Client, TestCase
from django.urls import reverse


class ServerUrlsTestCase(TestCase):
    """Test suite for the server views."""

    def setUp(self):
        """
        Set up the test case with a Django test client.
        """
        self.client = Client()

    def test_index_url(self):
        """
        Test the index URL.
        """
        response = self.client.get(reverse("fl_server:index"))
        self.assertEqual(response.status_code, 302)


    def test_dashboard_url(self):
        """
        Test the dashboard URL.
        """
        response = self.client.get(reverse("fl_server:dashboard"))
        self.assertEqual(response.status_code, 200)


    def test_projects_url(self):
        """
        Test the projects URL.
        """
        response = self.client.get(reverse("fl_server:projects"))
        self.assertEqual(response.status_code, 200)


    def test_server_stakeholders_renders_correct_template(self):
        """
        Test the stakeholders URL.
        """
        response = self.client.get(reverse("fl_server:stakeholders"))
        self.assertEqual(response.status_code, 200)


    def test_monitoring_url(self):
        """
        Test the monitoring URL.
        """
        response = self.client.get(reverse("fl_server:monitoring"))
        self.assertEqual(response.status_code, 200)


    def test_management_url(self):
        """
        Test the management URL.
        """
        response = self.client.get(reverse("fl_server:management"))
        self.assertEqual(response.status_code, 200)
