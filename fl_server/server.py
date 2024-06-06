import pandas as pd
import syft as sy

SYFT_VERSION = ">=0.8.2.b0,<0.9"


def launch_node(name, port):
    print(f"\n--- Démarrage du noeud {name} ---")
    node = sy.Orchestra.launch(
        name=name,
        port=port,
        local_db=True,
        dev_mode=True,
        reset=True,
    )
    return node


def register_user(node, email, password, name, institution, website):
    client = node.login(email=email, password=password)
    client.register(
        name=name,
        email=email,
        password=password,
        password_verify=password,
        institution=institution,
        website=website,
    )
    return client


def launch_and_register(
    name, port, email, password, user_name, institution, website
):
    node = launch_node(name, port)
    client = register_user(
        node, email, password, user_name, institution, website
    )
    return node, client


def launch_node_main():
    sy.requires(SYFT_VERSION)

    print(f"Version of PySyft : {sy.__version__}")

    node_humani, client_humani = launch_and_register(
        "do-humani",
        9000,
        "info@openmined.org",
        "changethis",
        "Jane Doe",
        "Caltech",
        "https://www.caltech.edu/",
    )
    node_epicura, client_epicura = launch_and_register(
        "do-epicura",
        9001,
        "info@openmined.org",
        "changethis",
        "Jane Doe",
        "Caltech",
        "https://www.caltech.edu/",
    )
    node_vivalia, client_vivalia = launch_and_register(
        "do-vivalia",
        9003,
        "info@openmined.org",
        "changethis",
        "Jane Doe",
        "Caltech",
        "https://www.caltech.edu/",
    )

    ds_client = node_humani.login(
        email="janedoe@caltech.edu",
        password="abc123",
    )

    data_subjects = client_humani.data_subject_registry.get_all()
    print(data_subjects)

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
    client_humani.upload_dataset(dataset)

    asset = ds_client.datasets[-1].assets["ages"]
    mock = asset.mock

    age_sum = mock["Age"].mean()
    print(age_sum)

    print("--- Destruction des Domain Servers ---")
    node_humani.land()
    node_epicura.land()
    node_vivalia.land()


if __name__ == "__main__":
    launch_node_main()
