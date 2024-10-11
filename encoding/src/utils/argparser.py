import os
import yaml
from typing import Dict, Union

def get_args(arg_dir: str) -> Dict[str, Union[float, str]]:
    """
    Gets relevant arguments from a yaml file.

    Parameters
    ----------
    arg_dir: str
        The path to the yaml file containing the arguments.
    
    Returns
    -------    
    args: Dict[str, Union[float, str]]
        The arguments in the form of a dictionary.
    """
    
    with open(arg_dir, "r") as f:
        args = yaml.safe_load(f)

    return args

def save_patch_args(
    output_dir: str, 
    args: Dict[str, Union[float, int]]
    ) -> None:

    """
    Saves the arguments as a yaml file.

    Parameters
    ----------
    output_dir: str
        The path to the saved patches.
    
    args: Dict[str, Union[float, str]]
        The arguments to save.
    """
    
    os.makedirs(output_dir, exist_ok=True)

    output_dir = os.path.join(output_dir, f"patch-config.yaml")

    with open(output_dir, "w") as f:
        yaml.dump(args, f)