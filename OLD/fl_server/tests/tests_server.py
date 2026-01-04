import pandas as pd
import syft as sy
from django.test import TestCase

from fl_server.server import land_node, launch_and_register, login

SYFT_VERSION = ">=0.8.7.b10,<0.9"


class ServerTestCase(TestCase):
    """
    A Django TestCase for testing the server functionality.
    """

    def setUp(self):
        """
        Set up the test case by launching and registering nodes.
        """
        sy.requires(SYFT_VERSION)
        self.node_humani, self.client_humani = launch_and_register(
            "do-humani",
            9000,
            "info@openmined.org",
            "changethis",
            "Jane Doe",
            "Caltech",
            "https://www.caltech.edu/",
        )
        self.node_epicura, self.client_epicura = launch_and_register(
            "do-epicura",
            9001,
            "info@openmined.org",
            "changethis",
            "Jane Doe",
            "Caltech",
            "https://www.caltech.edu/",
        )
        self.node_vivalia, self.client_vivalia = launch_and_register(
            "do-vivalia",
            9003,
            "info@openmined.org",
            "changethis",
            "Jane Doe",
            "Caltech",
            "https://www.caltech.edu/",
        )

    def test_launch_node_main(self):
        """
        Test the main node launch functionality.
        """
        data_subjects = self.client_humani.data_subject_registry.get_all()
        self.assertIsNotNone(data_subjects)

        dataset = sy.Dataset(
            name="usa-mock-data",
            description="Dataset of ages",
            asset_list=[
                sy.Asset(
                    name="ages",
                    data=pd.DataFrame(
                        {
                            "Patient_ID": ["011", "015", "022", "034", "044"],
                            "Age": [40, 39, 35, 60, 25],
                        },
                    ),
                    mock=pd.DataFrame(
                        {
                            "Patient_ID": ["1", "2", "3", "4", "5"],
                            "Age": [50, 49, 45, 70, 35],
                        },
                    ),
                    mock_is_real=False,
                ),
            ],
        )
        self.client_humani.upload_dataset(dataset)

        ds_client = login(
            "node_humani",
            "janedoe@caltech.edu",
            "abc123",
        )

        asset = ds_client.datasets[-1].assets["ages"]
        mock = asset.mock

        age_sum = mock["Age"].mean()
        self.assertIsNotNone(age_sum)

    def tearDown(self):
        """
        Clean up the test case by landing the nodes.
        """
        land_node(self.node_humani)
        land_node(self.node_epicura)
        land_node(self.node_vivalia)
