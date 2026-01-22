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

# --- Constants ---
LIST_SECTIONS = frozenset({"Metadata", "Candidate", "Target"})
DICT_SECTIONS = frozenset({"Thresholds", "Folder", "Options"})
KEY_VALUE_SEPARATOR = " = "


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Reads a custom configuration file.")
    parser.add_argument("--config_path", help="Path to the configuration file")
    return parser.parse_args()


def _create_default_config() -> Dict[str, Any]:
    """Create and return a default configuration dictionary."""
    return {
        "Metadata": [],
        "Candidate": [],
        "Target": [],
        "Exclude": {},
        "Thresholds": {},
        "Folder": {},
        "Options": {},
    }


def _is_section_header(line: str) -> bool:
    """Check if a line is a section header."""
    return line.startswith("[") and line.endswith("]")


def _is_comment_or_empty(line: str) -> bool:
    """Check if a line is a comment or empty."""
    return line.startswith("#") or line == ""


def _parse_key_value(line: str, lineno: int, section: str) -> tuple[str, str]:
    """Parse a key-value pair from a line.

    Args:
        line: The line to parse
        lineno: Line number for error reporting
        section: Current section name for error reporting

    Returns:
        Tuple of (key, value)

    Raises:
        ValueError: If the line is malformed
    """
    if KEY_VALUE_SEPARATOR not in line:
        raise ValueError(
            f"Malformed line at line {lineno} in section '{section}': '{line}'"
        )
    key, value = line.split(KEY_VALUE_SEPARATOR, 1)
    value = value.split("#")[0].strip()  # Remove inline comments
    return key.strip(), value


def _process_line(
    line: str, lineno: int, section: str, config: Dict[str, Any]
) -> None:
    """Process a single configuration line.

    Args:
        line: The stripped line content
        lineno: Line number for error reporting
        section: Current section name
        config: Configuration dictionary to update
    """
    if section in LIST_SECTIONS:
        config[section].append(line)
    elif section in DICT_SECTIONS:
        key, value = _parse_key_value(line, lineno, section)
        config[section][key] = value
    else:
        # Unknown section goes to Exclude
        if section not in config["Exclude"]:
            config["Exclude"][section] = []
        config["Exclude"][section].append(line)


def read_config(path: str) -> Dict[str, Any]:
    """Read a configuration file with multiline sections.

    Args:
        path: Path to the configuration file

    Returns:
        Dictionary with sections as keys and associated values

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If a configuration line is malformed
    """
    config = _create_default_config()
    section = ""

    try:
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()

                if _is_section_header(line):
                    section = line[1:-1]
                    continue

                if _is_comment_or_empty(line):
                    continue

                _process_line(line, lineno, section, config)

    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {path}")

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