import pandas as pd
import syft as sy

SYFT_VERSION = ">=0.8.2.b0,<0.9"
sy.requires(SYFT_VERSION)
print(f"Version of PySyft : {sy.__version__}")


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


def land_node(node):
    print("--- Destruction des Domain Servers ---")
    node.land()
