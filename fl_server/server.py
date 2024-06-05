import pandas as pd
import syft as sy

SYFT_VERSION = ">=0.8.6,<0.9"


def load_secrets():
    """
    Loads secrets from .env file.
    """
    from dotenv import load_dotenv
    import os

    load_dotenv()

    email = os.getenv("EMAIL")
    password = os.getenv("PASSWORD")


def launch_node(name, port, email, password):
    """
    Launches a node, registers a user, uploads a dataset, calculates the mean age,
    and then destroys the node.

    Args:
        name (str): Name of the node.
        port (int): Port number for the node.
        email (str): User email.
        password (str): User password.
    """

    print(f"\n--- Démarrage du noeud {name} ---")
    node = sy.orchestra.launch(
        name=f"do-{name}",
        port=port,
        local_db=True,
        dev_mode=True,
        reset=True,
    )
    client = node.login(
        email=email,
        password=password,
    )
    client.register(
        name="Jane Doe",
        email=email,
        password=password,
        password_verify=password,
        institution="Caltech",
        website="https://www.caltech.edu/",
    )

    ds_client = node.login(
        email=email,
        password=password,
    )

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
    client.upload_dataset(dataset)

    asset = ds_client.datasets[-1].assets["ages"]
    mock = asset.mock

    age_sum = mock["Age"].mean()
    print(age_sum)

    print(f"--- Destruction du Domain Server {name} ---")
    node.land()


def launch_nodes():
    """
    Launches three nodes: Humani, Epicura, and Vivalia.
    """
    email, password = load_secrets()
    nodes = [
        {"name": "Humani", "port": 9000},
        {"name": "Epicura", "port": 9001},
        {"name": "Vivalia", "port": 9003},
    ]
    for node in nodes:
        launch_node(node["name"], node["port"], email, password)


if __name__ == "__main__":
    sy.requires(SYFT_VERSION)
    print(f"Version of PySyft : {sy.__version__}")
    launch_nodes()
