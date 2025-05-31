from datetime import datetime
from typing import Dict

import yaml
import logging
from semiauto_classification.logger import configure_logger, section

configure_logger()
logger = logging.getLogger("utils")

def load_yaml(file_path: str) -> Dict:
    """
    Load YAML file into a dictionary.

    Args:
        file_path (str): Path to YAML file

    Returns:
        Dict: Loaded YAML content
    """
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Error loading YAML file {file_path}: {str(e)}")
        raise

def update_intel_yaml(intel_path: str, updates: Dict) -> None:
    """
    Update the intel.yaml file with new information.

    Args:
        intel_path (str): Path to intel.yaml file
        updates (Dict): Dictionary of updates to apply
    """
    try:
        # Load existing intel
        intel = load_yaml(intel_path)

        # Update with new information
        intel.update(updates)

        # Add processed timestamp
        intel['processed_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Write back to file
        with open(intel_path, 'w') as file:
            yaml.dump(intel, file, default_flow_style=False)

        logger.info(f"Updated intel.yaml at {intel_path}")
    except Exception as e:
        logger.error(f"Error updating intel.yaml: {str(e)}")
        raise