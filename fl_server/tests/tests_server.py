import os
import pandas as pd
import syft as sy

from django.test import TestCase
from unittest.mock import patch, MagicMock
from fl_server.server import launch_node, launch_nodes, load_secrets


class TestYourModule(TestCase):

    @patch('os.getenv')
    @patch('dotenv.load_dotenv')
    def test_load_secrets(self, mock_load_dotenv, mock_getenv):
        mock_getenv.side_effect = ['test_email', 'test_password']
        email, password = load_secrets()
        mock_load_dotenv.assert_called_once()
        self.assertEqual(email, 'test_email')
        self.assertEqual(password, 'test_password')

    @patch('your_module.sy.orchestra.launch')
    def test_launch_node(self, mock_launch):
        mock_node = MagicMock()
        mock_launch.return_value = mock_node
        launch_node('test', 9000, 'test_email', 'test_password')
        mock_launch.assert_called_once_with(name='do-test', port=9000, local_db=True, dev_mode=True, reset=True)

    @patch('fl_server.server.load_secrets')
    @patch('fl_server.server.launch_node')
    def test_launch_nodes(self, mock_launch_node, mock_load_secrets):
        mock_load_secrets.return_value = ('test_email', 'test_password')
        launch_nodes()
        self.assertEqual(mock_launch_node.call_count, 3)
