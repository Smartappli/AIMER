import os
import pandas as pd
import pytest
import syft as sy
from unittest.mock import patch, MagicMock

# Import your functions here
from fl_server.server import load_secrets, launch_node, launch_nodes


def test_load_secrets():
    with patch('os.getenv') as mock_getenv:
        mock_getenv.side_effect = ['test_email', 'test_password']
        email, password = load_secrets()
        assert email == 'test_email'
        assert password == 'test_password'


@patch('syft.orchestra.launch')
@patch('fl_server.server.load_secrets')
def test_launch_node(mock_load_secrets, mock_launch):
    mock_load_secrets.return_value = ('test_email', 'test_password')
    mock_node = MagicMock()
    mock_launch.return_value = mock_node
    mock_client = MagicMock()
    mock_node.login.return_value = mock_client
    launch_node('test_node', 9000, 'test_email', 'test_password')
    assert mock_node.login.call_count == 2


@patch('fl_server.server.launch_node')
@patch('fl_server.server.load_secrets')
def test_launch_nodes(mock_load_secrets, mock_launch_node):
    mock_load_secrets.return_value = ('test_email', 'test_password')
    launch_nodes()
    assert mock_launch_node.call_count == 3
