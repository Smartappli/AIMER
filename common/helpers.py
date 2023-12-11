import pathlib
import json

from typing import Dict, List, Any

def set_config_file(config_type: str) -> str:
    # set the config file name
    module_path = pathlib.Path.cwd()
    config_file = f'{module_path}/config_{config_type}.json'

    return config_file


def read_config(config_path: str) -> Dict[str, Any]:
    """
    Read a JSON configuration file to set up an aggregator
    :param config_path:
    :return: Dict[str,Any] - configs
    """
    with open(config_path) as json_file:
        config = json.load(json_file)
    return config

