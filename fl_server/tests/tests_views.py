from django.test import Client, TestCase
from django.urls import reverse


class ServerViewsTestCase(TestCase):
    """Test suite for the server views."""

    def setUp(self):
        """
        Set up the test case with a Django test client.
        """
        self.client = Client()

    def test_index_redirects_to_dashboard(self):
        """
        Test that the index view redirects to the dashboard view.
        """
        response = self.client.get(reverse("fl_server:index"))
        self.assertRedirects(response, "/server/dashboard/")

    def test_dashboard_renders_correct_template(self):
        """
        Test that the dashboard view renders the correct template with the correct context.
        """
        response = self.client.get(reverse("fl_server:dashboard"))
        self.assertTemplateUsed(response, "server/server_dashboard.html")
        self.assertEqual(response.context["section"], "Dashboard")
        self.assertListEqual(
            response.context["logo"],
            ["share", "hospital", "data", "cpu", "gpu"],
        )

    def test_server_projects_renders_correct_template(self):
        """
        Test that the server projects view renders the correct template with the correct context.
        """
        response = self.client.get(reverse("fl_server:projects"))
        self.assertTemplateUsed(response, "server/server_projects.html")
        self.assertEqual(response.context["section"], "projects")
        self.assertListEqual(
            response.context["logo"],
            ["share", "hospital", "data", "cpu", "gpu"],
        )

    def test_server_stakeholders_renders_correct_template(self):
        """
        Test that the server stakeholders view renders the correct template with the correct context.
        """
        response = self.client.get(reverse("fl_server:stakeholders"))
        self.assertTemplateUsed(response, "server/server_stakeholders.html")
        self.assertEqual(response.context["section"], "stakeholders")
        self.assertListEqual(
            response.context["logo"],
            ["share", "hospital", "data", "cpu", "gpu"],
        )

    def test_server_monitoring_renders_correct_template(self):
        """
        Test that the server monitoring view renders the correct template with the correct context.
        """
        response = self.client.get(reverse("fl_server:monitoring"))
        self.assertTemplateUsed(response, "server/server_monitoring.html")
        self.assertEqual(response.context["section"], "monitoring")
        self.assertListEqual(
            response.context["logo"],
            ["share", "hospital", "data", "cpu", "gpu"],
        )

    def test_server_management_renders_correct_template(self):
        """
        Test that the server management view renders the correct template with the correct context.
        """
        response = self.client.get(reverse("fl_server:management"))
        self.assertTemplateUsed(response, "server/server_management.html")
        self.assertEqual(response.context["section"], "management")
        self.assertListEqual(
            response.context["logo"],
            ["share", "hospital", "data", "cpu", "gpu"],
        )
