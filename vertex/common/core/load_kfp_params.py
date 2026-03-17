import yaml
from pathlib import Path
from typing import Dict, Any
from demo.vertex.common.core.settings import get_settings

def load_kfp_params(pipeline_folder: str, filename: str) -> Dict[str, Any]:
    """
    Dynamically loads KFP arguments from YAML files in the parameters/ directory.
   
    Args:
        pipeline_folder (str): Subdirectory name inside parameters/.
        filename (str): Name of the YAML file (can be provided without extension).
       
    Returns:
        (dict) A dictionary with the parameters to be injected into the pipeline.
    """    

    settings = get_settings()

    # Construct absolute path using BASE_DIR from Settings
    file = settings.BASE_DIR/ "vertex" / "parameters" / pipeline_folder / f"{filename}.yaml"

    # YAML/YML support
    extensions = [".yaml", ".yml"]
    file_to_load = next(
        (file.with_suffix(ext) for ext in extensions if file.with_suffix(ext).exists()),
        None
    )

    if not file_to_load:
        raise FileNotFoundError(
            f"❌ Pipeline parameters file not found: '{filename}' in: parameters/{pipeline_folder}/"
        )

    try:
        with open(file_to_load, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)
           
            # If the file is empty, yaml.safe_load returns None; we ensure a dict
            return params if params is not None else {}
           
    except Exception as e:
        raise RuntimeError(f"❌ Error parsing YAML at {file_to_load}: {str(e)}")    