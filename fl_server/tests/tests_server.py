from django.test import TestCase
from unittest.mock import patch, MagicMock

# Import your functions here
from fl_server.server import load_secrets, launch_node, launch_nodes


class TestServer(TestCase):

    @patch('os.getenv')
    def test_load_secrets(self, mock_getenv):
        mock_getenv.side_effect = ['test_email', 'test_password']
        email, password = load_secrets()
        self.assertEqual(email, 'test_email')
        self.assertEqual(password, 'test_password')

    @patch('syft.orchestra.launch')
    @patch('fl_server.server.load_secrets')
    def test_launch_node(self, mock_load_secrets, mock_launch):
        mock_load_secrets.return_value = ('test_email', 'test_password')
        mock_node = MagicMock()
        mock_launch.return_value = mock_node
        mock_client = MagicMock()
        mock_node.login.return_value = mock_client
        launch_node('test_node', 9000, 'test_email', 'test_password')
        self.assertEqual(mock_node.login.call_count, 2)

    @patch('fl_server.server.launch_node')
    @patch('fl_server.server.load_secrets')
    def test_launch_nodes(self, mock_load_secrets, mock_launch_node):
        mock_load_secrets.return_value = ('test_email', 'test_password')
        launch_nodes()
        self.assertEqual(mock_launch_node.call_count, 3)