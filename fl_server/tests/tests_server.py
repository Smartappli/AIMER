import unittest
from unittest.mock import patch, MagicMock


class TestNodeLaunch(unittest.TestCase):
    @patch("syft.Orchestra.launch")
    def test_launch_node(self, mock_launch):
        from fl_server.server import (
            launch_node,
        )  # Assuming the refactored code is in fl_server.server.py

        mock_node = MagicMock()
        mock_launch.return_value = mock_node

        node = launch_node("do-humani", 9000)

        mock_launch.assert_called_once_with(
            name="do-humani",
            port=9000,
            local_db=True,
            dev_mode=True,
            reset=True,
        )
        self.assertEqual(node, mock_node)

    @patch("syft.Orchestra.login")
    def test_register_user(self, mock_login):
        from fl_server.server import (
            register_user,
        )  # Assuming the refactored code is in fl_server.server.py

        mock_client = MagicMock()
        mock_login.return_value = mock_client

        node = MagicMock()
        client = register_user(
            node,
            "info@openmined.org",
            "changethis",
            "Jane Doe",
            "Caltech",
            "https://www.caltech.edu/",
        )

        mock_login.assert_called_once_with(
            email="info@openmined.org", password="changethis"
        )
        mock_client.register.assert_called_once_with(
            name="Jane Doe",
            email="info@openmined.org",
            password="changethis",
            password_verify="changethis",
            institution="Caltech",
            website="https://www.caltech.edu/",
        )
        self.assertEqual(client, mock_client)


if __name__ == "__main__":
    unittest.main()
