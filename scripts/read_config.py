"""
read_config.py
--------------
Read custom configuration files with multiple sections.

Author: Guillaume Lemaire
License: MIT

Examples:
    >>> read_config("config.txt")
    {'Metadata': ['gene_id', 'gene_name'], 'Markers': ['gene1', 'gene3', 'gene4']}
"""

from typing import Dict, Any
import argparse
import sys

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Reads a custom configuration file.")
    parser.add_argument("--config_path", help="Path to the configuration file")
    return parser.parse_args()

def read_config(path: str) -> Dict[str, Any]:
    """Reads a configuration file with multiline sections.

    Args:
        path (str): Path to the configuration file
    Returns:
        dict: Dictionary with sections as keys and associated values
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If a configuration line is malformed or section unknown
    """
    config = dict([["Metadata",[]], ["Candidate",[]], ["Markers",[]], ["Exclude",[]], 
                   ["Thresholds",dict()], ["Folder",dict()], ["Options",dict()]])
    section = None
    list_sections = {"Metadata", "Candidate", "Markers", "Exclude"}
    dict_sections = {"Thresholds", "Folder", "Options"}

    try:
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if line.startswith("[") and line.endswith("]"):
                    section = line[1:-1]
                    if section not in list_sections | dict_sections:
                        raise ValueError(f"Unknown section '{section}' at line {lineno}")
                    continue

                if section in list_sections:
                    if line != "":
                        config[section].append(line)
                else:
                    if line == "" or line == "\n":
                        continue
                    elif " = " not in line:
                        raise ValueError(f"Malformed line at line {lineno} in section '{section}': '{line}'")
                    key, value = line.split(" = ", 1)
                    value = value.split("#")[0].strip()
                    config[section][key.strip()] = value
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {path}")
    except Exception as e:
        raise e
    return config

if __name__ == "__main__":
    args = parse_args()
    try:
        config = read_config(args.config_path)
        print("Configuration read:")
        for section, values in config.items():
            print(f"[{section}]")
            print(values)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)