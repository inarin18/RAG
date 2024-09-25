import os
import yaml
import logging


def setup_logging(log_level):
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)