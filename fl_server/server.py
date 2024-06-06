import pandas as pd
import syft as sy

SYFT_VERSION = ">=0.8.2.b0,<0.9"

# Ensure the correct version of PySyft is installed
sy.requires(SYFT_VERSION)
print(f"Version of PySyft : {sy.__version__}")

def launch_node(name, port):
    """
    Launch a new node with the given name and port.

    Args:
        name (str): The name of the node.
        port (int): The port number for the node.

    Returns:
        node: The launched node.
    """
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
    """
    Register a new user for the given node.

    Args:
        node: The node to register the user for.
        email (str): The email of the user.
        password (str): The password of the user.
        name (str): The name of the user.
        institution (str): The institution of the user.
        website (str): The website of the user.

    Returns:
        client: The registered client.
    """
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

def launch_and_register(name, port, email, password, user_name, institution, website):
    """
    Launch a new node and register a new user for it.

    Args:
        name (str): The name of the node.
        port (int): The port number for the node.
        email (str): The email of the user.
        password (str): The password of the user.
        user_name (str): The name of the user.
        institution (str): The institution of the user.
        website (str): The website of the user.

    Returns:
        node: The launched node.
        client: The registered client.
    """
    node = launch_node(name, port)
    client = register_user(
        node, email, password, user_name, institution, website
    )
    return node, client

def login(node, login_email, login_password):
    """
    Log in to the given node with the provided email and password.

    Args:
        node: The node to log in to.
        login_email (str): The email to use for logging in.
        login_password (str): The password to use for logging in.

    Returns:
        client: The client returned after successful login.
    """
    return node.login(
        email=login_email,
        password=login_password,
    )

def land_node(node):
    """
    Destroy the given node.

    Args:
        node: The node to destroy.
    """
    print("--- Destruction des Domain Servers ---")
    node.land()
