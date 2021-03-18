import toml
import os

CONFIG_FILE_DIR = "rave/configs"


def parse_config_file(config_file: str):
    """
    Args:
        config_file: path to a config file
    """
    if os.path.isfile(config_file):
        return toml.load(config_file)
    elif os.path.isfile(os.path.join(CONFIG_FILE_DIR, config_file)):
        return toml.load(os.path.join(CONFIG_FILE_DIR, config_file))
    else:
        raise IOError("Couldn't find the config file", config_file)


def write_config_to_file(config: dict, write_to: str):
    write_to_path = os.path.join(CONFIG_FILE_DIR, write_to)
    return toml.dump(config, write_to_path, encoder=toml.TomlNumpyEncoder)
