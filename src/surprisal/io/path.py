from dataclasses import dataclass


@dataclass
class PathConfig:
    """Singleton access to paths of important files and directories."""

    # Project root directory for all data.
    # NOTE: '..' is correct assuming CWD is directly under project root. YMMV.
    # NOTE: Use relative paths to be independent of encompassing filesystem.
    root_dir: str = ".."

    # Non-code data directory.
    data_dir: str = "${root_dir}/data"

    # Input/resource data directory.
    resources_dir: str = "${data_dir}/resources"

    # Output/results data directory.
    results_dir: str = "${data_dir}/results"

    # ConceptNet KB directory, in the same format as the ACCORD post-processing.
    concept_net_dir: str = "${resources_dir}/ConceptNet"

    # Top-level result directory for experiment 1.
    experiment1_dir: str = "${results_dir}/experiment1"
