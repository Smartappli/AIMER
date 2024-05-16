import pandas as pd
import syft as sy

# from syft import autocache

SYFT_VERSION = ">=0.8.2.b0,<0.9"
package_string = f'"syft{SYFT_VERSION}"'
verbose = True


def launch_node():
    """
    Launches three nodes: Humani, Epicura, and Vivalia, registers a user,
    uploads a dataset, calculates the mean age, and then destroys the nodes.
    """

    sy.requires(SYFT_VERSION)

    print(f"Version of PySyft : {sy.__version__}")

    print("\n--- Démarrage du noeud Humani ---")
    node_humani = sy.Orchestra.launch(
        name="do-humani", port=9000, local_db=True, dev_mode=True, reset=True
    )
    root_domain_humani_client = node_humani.login(
        email="info@openmined.org", password="changethis"
    )
    root_domain_humani_client.register(
        name="Jane Doe",
        email="janedoe@caltech.edu",
        password="abc123",
        password_verify="abc123",
        institution="Caltech",
        website="https://www.caltech.edu/",
    )

    print("\n--- Démarrage du noeud Epicura ---")
    node_epicura = sy.Orchestra.launch(
        name="do-epicura", port=9001, local_db=True, dev_mode=True, reset=True
    )
    root_domain_epicura_client = node_epicura.login(
        email="info@openmined.org", password="changethis"
    )
    root_domain_epicura_client.register(
        name="Jane Doe",
        email="janedoe@caltech.edu",
        password="abc123",
        password_verify="abc123",
        institution="Caltech",
        website="https://www.caltech.edu/",
    )

    print("\n--- Démarrage du noeud Vivalia ---")
    node_vivalia = sy.Orchestra.launch(
        name="do-vivalia", port=9003, local_db=True, dev_mode=True, reset=True
    )
    root_domain_vivalia_client = node_vivalia.login(
        email="info@openmined.org", password="changethis"
    )
    root_domain_vivalia_client.register(
        name="Jane Doe",
        email="janedoe@caltech.edu",
        password="abc123",
        password_verify="abc123",
        institution="Caltech",
        website="https://www.caltech.edu/",
    )

    ds_client = node_humani.login(
        email="janedoe@caltech.edu",
        password="abc123")

    data_subjects = root_domain_humani_client.data_subject_registry.get_all()
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
                    }
                ),
                mock=pd.DataFrame(
                    {
                        "Patient_ID": ["1", "2", "3", "4", "5"],
                        "Age": [50, 49, 45, 70, 35],
                    }
                ),
                mock_is_real=False,
            )
        ],
    )
    root_domain_humani_client.upload_dataset(dataset)

    asset = ds_client.datasets[-1].assets["ages"]
    mock = asset.mock

    age_sum = mock["Age"].mean()
    print(age_sum)

    print("--- Destruction des Domain Servers ---")
    node_humani.land()
    node_epicura.land()
    node_vivalia.land()


if __name__ == "__main__":
    launch_node()
