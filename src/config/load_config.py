# This file is originally developed by https://github.com/mohsenh17 in:
# https://github.com/mohsenh17/bulkGeneFormer/tree/main/bulkGeneFormer/config
# It is reused and further developed as part of this project.

import yaml
from pathlib import Path
import os
from typing import Dict, Any

def construct_training_paths(cfg):
    """Constructs training paths based on the configuration.
    This function modifies the configuration dictionary to include paths for checkpoints and final model directories.
    - Parameters:
        cfg (dict): The configuration dictionary containing model name, version, and output base.
    - Returns:
        dict: The modified configuration dictionary with added paths for checkpoints and final model.
    """
    name = cfg['model_name']
    version = cfg['version']
    base = cfg['output_base']
    cfg['checkpoint_dir'] = f"{base}/{name}_{version}/checkpoints"
    cfg['final_model_dir'] = f"{base}/{name}_{version}/final_model"
    return cfg


def resolve_paths(d: Dict[str, Any], root: Path) -> Dict[str, Any]:
    """
    Recursively resolves all paths in a nested dictionary.
    """
    resolved = {}
    for key, value in d.items():
        if isinstance(value, dict):
            resolved[key] = resolve_paths(value, root)
        else:
            resolved[key] = (root / Path(value)).resolve()
    return resolved

def load_config(config_path="configs/config.yaml"):
    """
    Loads the configuration from a YAML file and resolves paths.
    - Parameters:
        config_path (str): The path to the configuration file.
    - Returns:
        dict: A dictionary containing the resolved configuration.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Expand environment variables and resolve path
    project_root = Path(os.path.expandvars(config["project_root"])).resolve()

    config['training'] = construct_training_paths(config['training'])
    
    resolved_paths = resolve_paths(config["paths"], project_root)

    return {
        "project_root": project_root,
        "paths": resolved_paths,
        "training": config["training"],
    }

if __name__ == "__main__":
    config = load_config()
    print("Configuration loaded successfully.")
    print(f"Project root: {config['project_root']}")
    print("Paths:")
    for key, path in config['paths'].items():
        print(f"  {key}: {path}")
    
    for key, value in config['training'].items():
        print(f"  {key}: {value}")