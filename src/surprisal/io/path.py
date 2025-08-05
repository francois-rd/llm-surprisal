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

    # ACCORD dataset directory, in the same format as the original ACCORD paper.
    accord_dir: str = "${resources_dir}/ACCORD"

    # ACCORD reductions/surface forms dataset file, in the same format as the
    # original ACCORD paper.
    accord_reductions_file: str = "${accord_dir}/reductions.csv"

    # ACCORD base CSQA dataset file, in the same format as the original ACCORD paper.
    accord_csqa_file: str = "${accord_dir}/csqa_base.jsonl"

    # ACCORD baseline dataset file, in the same format as the original ACCORD paper.
    accord_baseline_file: str = "${accord_dir}/0.jsonl"

    # ACCORD tree size 1 dataset file, in the same format as the original ACCORD paper.
    accord_tree_size_1_file: str = "${accord_dir}/1.jsonl"

    # ACCORD tree size 2 dataset file, in the same format as the original ACCORD paper.
    accord_tree_size_2_file: str = "${accord_dir}/2.jsonl"

    # ACCORD tree size 3 dataset file, in the same format as the original ACCORD paper.
    accord_tree_size_3_file: str = "${accord_dir}/3.jsonl"

    # ACCORD tree size 4 dataset file, in the same format as the original ACCORD paper.
    accord_tree_size_4_file: str = "${accord_dir}/4.jsonl"

    # ACCORD tree size 5 dataset file, in the same format as the original ACCORD paper.
    accord_tree_size_5_file: str = "${accord_dir}/5.jsonl"

    # Top-level result directory for experiment 1.
    experiment1_dir: str = "${results_dir}/experiment1"

    # Top-level result directory for experiment 2.
    experiment2_dir: str = "${results_dir}/experiment2"
