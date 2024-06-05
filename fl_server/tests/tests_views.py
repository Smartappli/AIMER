from django.test import TestCase
from django.urls import reverse


class ServerViewsTestCase(TestCase):
    """Test suite for the server views."""

    def test_index_redirects_to_dashboard(self):
        """
        Test that the index view redirects to the dashboard view.
        """
        response = self.client.get(reverse("index"))
        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, "/server/dashboard/")

    def test_dashboard_renders_correct_template(self):
        """
        Test that the dashboard view renders the correct template with the correct context.
        """
        response = self.client.get(reverse("dashboard"))
        self.assertEqual(response.status_code, 200)
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
        response = self.client.get(reverse("server_projects"))
        self.assertEqual(response.status_code, 200)
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
        response = self.client.get(reverse("server_stakeholders"))
        self.assertEqual(response.status_code, 200)
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
        response = self.client.get(reverse("server_monitoring"))
        self.assertEqual(response.status_code, 200)
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
        response = self.client.get(reverse("server_management"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "server/server_management.html")
        self.assertEqual(response.context["section"], "management")
        self.assertListEqual(
            response.context["logo"],
            ["share", "hospital", "data", "cpu", "gpu"],
        )
