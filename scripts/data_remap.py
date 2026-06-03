"""
Data Remapping Script for dVRK Multi-modal Data Collection

This script remaps already organized data from one sequence to another based on
a remap configuration file. It copies data folders and updates metadata accordingly.

Input: Organized data folder (e.g., raw_data) with sequential numbered folders
Output: Remapped data folder with new sequence based on remap_data_note.json

Supports both simple indices and nested paths for flexible data organization.

IMPORTANT: The remap JSON file supports // style comments, which will be
automatically stripped during parsing.
"""

import os
import sys
import json
import re
import shutil
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
from tqdm import tqdm


@dataclass
class RemapConfig:
    """
    Configuration for data remapping script.

    All configuration parameters are at the top level for simplicity.
    """
    workspace: str
    input_folder: str
    output_folder: str
    remap_config: str  # Path to remap_data_note.json


def load_remap_mapping(remap_file: Path) -> Dict[str, str]:
    """
    Load the remap mapping from JSON file.

    Args:
        remap_file: Path to remap_data_note.json

    Returns:
        Dictionary mapping output folder indices to input folder paths

    Note:
        The JSON file should have the format:
        {
          "<output_idx>": <input_idx_or_path>,
          ...
        }

        This means: "what goes into output folder X" <- "comes from input folder Y"

        Supports multiple formats:
        - Simple index: "0": 113 -> output/0 gets data from input/113
        - Nested paths: "test/0": 5 -> output/test/0 gets data from input/5
        - Integer values: "0": 10 -> output/0 gets data from input/10

        This function normalizes all keys and values to strings.

        Comments (// style) are supported and will be stripped before parsing.
    """
    if not remap_file.exists():
        raise FileNotFoundError(f"Remap configuration file not found: {remap_file}")

    try:
        with open(remap_file, 'r') as f:
            # Read the file content and strip out comments
            # Support // style comments (both at beginning of line and inline)
            lines = []
            for line in f:
                # Find the position of // comment marker
                comment_pos = line.find('//')
                if comment_pos != -1:
                    # Strip everything after // (including the //)
                    line = line[:comment_pos]
                # Only add non-empty lines (after stripping whitespace)
                if line.strip():
                    lines.append(line)

            # Join the cleaned lines and parse as JSON
            cleaned_content = '\n'.join(lines)

            # Remove trailing commas before closing braces/brackets
            # This handles common JSON-with-comments formatting issues
            cleaned_content = re.sub(r',(\s*[}\]])', r'\1', cleaned_content)

            raw_mapping = json.loads(cleaned_content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in remap file {remap_file}: {e}")

    # Normalize the mapping: string keys, string values
    # This handles:
    # - "0": 5 -> "0": "5"
    # - "0": "5" -> "0": "5"
    # - "0": "test/5" -> "0": "test/5"
    normalized_mapping = {}
    for key, value in raw_mapping.items():
        try:
            # Convert both key and value to strings
            str_key = str(key)
            str_value = str(value)
            normalized_mapping[str_key] = str_value
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid mapping entry {key}: {value} - {e}")

    return normalized_mapping


def load_organization_note(note_file: Path) -> Dict[str, Any]:
    """
    Load the data_organization_note.json file.

    Args:
        note_file: Path to data_organization_note.json

    Returns:
        Dictionary containing organization metadata for all datasets

    Note:
        Returns empty dict if file doesn't exist (will be created during remap).
    """
    if not note_file.exists():
        return {}

    try:
        with open(note_file, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Warning: Could not load organization note from {note_file}: {e}")
        return {}


def save_organization_note(note_file: Path, data: Dict[str, Any]):
    """
    Save the data_organization_note.json file with proper formatting.

    Args:
        note_file: Path to data_organization_note.json
        data: Dictionary to save

    Note:
        Saves with smart key sorting:
        - Numeric keys are sorted numerically (0, 1, 2, ...)
        - String/path keys are sorted alphabetically
        Pretty formatting with 2-space indentation for readability.
    """
    # Create parent directory if needed
    note_file.parent.mkdir(parents=True, exist_ok=True)

    # Sort keys intelligently: try numeric first, fallback to string
    def get_sort_key(key):
        try:
            # Try to parse as integer for numeric sorting
            return (0, int(key), "")
        except ValueError:
            # Fallback to string sorting for paths like "test/0"
            return (1, 0, key)

    sorted_data = {k: data[k] for k in sorted(data.keys(), key=get_sort_key)}

    with open(note_file, 'w') as f:
        json.dump(sorted_data, f, indent=2)


def copy_dataset_folder(src_folder: Path, dst_folder: Path) -> bool:
    """
    Copy entire dataset folder from source to destination.

    Args:
        src_folder: Source dataset folder (e.g., raw_data/0)
        dst_folder: Destination dataset folder (e.g., remapped_data/5)

    Returns:
        True if successful, False otherwise

    Note:
        This copies the entire folder structure including:
        - image/ (all camera subfolders)
        - kinematic/
        - time_syn/
        - annotation/
        - camera_calibration/
        - hand_eye_calibration/
    """
    if not src_folder.exists():
        print(f"  Error: Source folder does not exist: {src_folder}")
        return False

    try:
        # Remove destination if it already exists to ensure clean copy
        if dst_folder.exists():
            shutil.rmtree(dst_folder)

        # Copy the entire directory tree
        shutil.copytree(src_folder, dst_folder)
        return True

    except Exception as e:
        print(f"  Error copying {src_folder} to {dst_folder}: {e}")
        return False


def update_metadata_entry(original_entry: Dict[str, Any],
                         new_path: str,
                         output_folder: Path) -> Dict[str, Any]:
    """
    Update metadata entry for the remapped dataset.

    Args:
        original_entry: Original metadata entry from input data_organization_note.json
        new_path: New path for the remapped dataset (can be simple index or nested path)
        output_folder: Output folder path

    Returns:
        Updated metadata entry with new paths

    Note:
        Preserves original_data_path and user_info from the source.
        Updates new_data_path and full_path to reflect the remapped location.

        Supports nested paths:
        - Simple index: new_path="5" -> output_folder/5
        - Nested path: new_path="test/5" -> output_folder/test/5
    """
    # Create a copy of the original entry
    updated_entry = original_entry.copy()

    # Update the new_data_path to reflect the remapped location
    # Supports both simple indices ("5") and nested paths ("test/5")
    # Format: <output_folder_name>/<new_path>
    try:
        new_data_rel = Path(output_folder.name) / new_path
    except:
        new_data_rel = Path("remapped_data") / new_path

    updated_entry["new_data_path"] = str(new_data_rel)

    # Update the full_path to reflect absolute path of remapped location
    if "full_path" not in updated_entry:
        updated_entry["full_path"] = {}

    # Keep the original path from the source (preserve provenance)
    # Update the new path to point to remapped location (supports nested paths)
    new_full_path = output_folder / new_path
    updated_entry["full_path"]["new"] = str(new_full_path.resolve())

    return updated_entry


def remap_data(config: RemapConfig) -> int:
    """
    Main data remapping function.

    Args:
        config: Remap configuration

    Returns:
        Number of datasets successfully remapped

    Note:
        This function:
        1. Loads the remap mapping from remap_data_note.json
        2. For each mapping entry, copies the dataset folder to new location
        3. Updates data_organization_note.json with remapped metadata
        4. Maintains all data integrity and metadata provenance
    """
    input_path = Path(config.input_folder)
    output_path = Path(config.output_folder)
    remap_file = Path(config.remap_config)

    # Validate input folder exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder not found: {input_path}")

    # Create output folder if needed
    output_path.mkdir(parents=True, exist_ok=True)

    # Load remap mapping
    print(f"Loading remap configuration from {remap_file}...")
    remap_mapping = load_remap_mapping(remap_file)

    if not remap_mapping:
        print("No remap mappings found in configuration file.")
        return 0

    print(f"Found {len(remap_mapping)} remap mapping(s)")
    print("\nRemap plan:")
    # Sort by output index (try to parse as int, fallback to string comparison)
    # Note: Keys are OUTPUT indices, values are INPUT indices
    def get_sort_key(item):
        try:
            return (0, int(item[0]))  # Numeric indices first
        except ValueError:
            return (1, item[0])  # String indices second

    for dst_path, src_idx in sorted(remap_mapping.items(), key=get_sort_key):
        print(f"  {input_path.name}/{src_idx} -> {output_path.name}/{dst_path}")
    print()

    # Load input organization note
    input_note_file = input_path / "data_organization_note.json"
    input_org_note = load_organization_note(input_note_file)

    # Prepare output organization note
    output_org_note = {}

    # Process each remap entry
    # Note: Keys in remap_mapping are OUTPUT indices, values are INPUT indices
    processed_count = 0
    failed_mappings = []

    for dst_path, src_idx in tqdm(remap_mapping.items(), desc="Remapping datasets"):
        # src_idx is where we READ from (input folder)
        # dst_path is where we WRITE to (output folder)
        src_folder = input_path / str(src_idx)
        # dst_path can be a simple index ("0") or a nested path ("test/0")
        dst_folder = output_path / dst_path

        print(f"\n[{processed_count + 1}/{len(remap_mapping)}] Remapping: {src_idx} -> {dst_path}")

        # Copy the dataset folder (creates parent directories if needed for nested paths)
        if copy_dataset_folder(src_folder, dst_folder):
            processed_count += 1

            # Update metadata if available in input organization note
            # Convert src_idx to string for dictionary lookup
            src_key = str(src_idx)
            if src_key in input_org_note:
                updated_entry = update_metadata_entry(
                    input_org_note[src_key],
                    dst_path,  # Now supports nested paths
                    output_path
                )
                # Use dst_path as the key in the output JSON (supports nested paths like "test/0")
                output_org_note[dst_path] = updated_entry
            else:
                print(f"  Warning: No metadata found for input index {src_idx}")
                # Create minimal metadata entry
                output_org_note[dst_path] = {
                    "original_data_path": f"(remapped from {input_path.name}/{src_idx})",
                    "new_data_path": f"{output_path.name}/{dst_path}",
                    "full_path": {
                        "original": str((input_path / str(src_idx)).resolve()),
                        "new": str((output_path / dst_path).resolve())
                    },
                    "user_info": {
                        "user_id": "",
                        "user_skill_level": {"dVRK": -1, "clinical": -1},
                        "user_description": ""
                    }
                }
        else:
            failed_mappings.append((dst_path, src_idx))

    # Save output organization note
    output_note_file = output_path / "data_organization_note.json"
    save_organization_note(output_note_file, output_org_note)

    # Print summary
    print(f"\n{'='*70}")
    print(f"Data remapping complete!")
    print(f"Successfully remapped: {processed_count}/{len(remap_mapping)} datasets")

    if failed_mappings:
        print(f"\nFailed mappings:")
        for dst_path, src_idx in failed_mappings:
            print(f"  {src_idx} -> {dst_path}")

    print(f"\nInput folder: {input_path}")
    print(f"Output folder: {output_path}")
    print(f"Output organization note: {output_note_file}")
    print(f"{'='*70}")

    return processed_count


# Configure Hydra
cs = ConfigStore.instance()
cs.store(name="data_remap", node=RemapConfig)

# Set config path - use absolute path for reliability
project_root = Path(__file__).resolve().parent.parent
p_config = project_root / 'config'

# Verify config path exists
if not p_config.exists():
    raise FileNotFoundError(f"Config directory not found: {p_config}")

@hydra.main(
    version_base=None,
    config_path=str(p_config),
    config_name="config_data_remap"
)
def main(cfg: DictConfig):
    """
    Main entry point for the data remapping script.

    Args:
        cfg: Hydra configuration object (DictConfig from OmegaConf)

    Note:
        This function is decorated with @hydra.main to enable Hydra-based
        configuration management. Configuration can be overridden from command line.
    """
    print("="*70)
    print("dVRK Multi-modal Data Remapping Script")
    print("="*70)
    print(f"Workspace: {cfg.workspace}")
    print(f"Input folder: {cfg.input_folder}")
    print(f"Output folder: {cfg.output_folder}")
    print(f"Remap config: {cfg.remap_config}")
    print("="*70)

    # Convert DictConfig to RemapConfig
    remap_config = RemapConfig(
        workspace=cfg.workspace,
        input_folder=cfg.input_folder,
        output_folder=cfg.output_folder,
        remap_config=cfg.remap_config
    )

    # Run the remapping
    try:
        processed = remap_data(remap_config)

        if processed > 0:
            print(f"\n✓ Successfully remapped {processed} dataset(s)")
        else:
            print(f"\n✗ No datasets were remapped")

    except Exception as e:
        print(f"\n✗ Error during data remapping: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
