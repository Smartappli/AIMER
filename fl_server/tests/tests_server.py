import os
from dotenv import load_dotenv
from unittest.mock import MagicMock, patch

import pandas as pd
import syft as sy
from django.test import TestCase

from fl_server.server import launch_node, launch_nodes, load_secrets

SYFT_VERSION = ">=0.8.6,<0.9"
sy.requires(SYFT_VERSION)


class TestYourModule(TestCase):
    """
    Test case for the module.
    """

    @patch("os.getenv")
    @patch("dotenv.load_dotenv")
    def test_load_secrets(self, mock_load_dotenv, mock_getenv):
        """
        Test case for the load_secrets function.
        """
        # Mock the return values of environment variables
        mock_getenv.side_effect = ["test_email", "test_password"]

        # Call the load_secrets function
        email, password = load_secrets()

        # Assert that load_dotenv was called once
        mock_load_dotenv.assert_called_once()

        # Assert that the returned email and password match the mocked values
        self.assertEqual(email, "test_email")
        self.assertEqual(password, "test_password")

    @patch("sy.orchestra.launch")
    def test_launch_node(self, mock_launch):
        """
        Test case for the launch_node function.
        """
        # Create a mock node
        mock_node = MagicMock()
        mock_launch.return_value = mock_node

        # Call the launch_node function with test parameters
        launch_node("test", 9000, "test_email", "test_password")

        # Assert that the launch function was called once with the specified parameters
        mock_launch.assert_called_once_with(
            name="do-test",
            port=9000,
            local_db=True,
            dev_mode=True,
            reset=True,
        )

    @patch("fl_server.server.load_secrets")
    @patch("fl_server.server.launch_node")
    def test_launch_nodes(self, mock_launch_node, mock_load_secrets):
        """
        Test case for the launch_nodes function.
        """
        # Mock the return values of load_secrets
        mock_load_secrets.return_value = ("test_email", "test_password")

        # Call the launch_nodes function
        launch_nodes()

        # Assert that the launch_node function was called three times
        self.assertEqual(mock_launch_node.call_count, 3)
