from dataclasses import dataclass


@dataclass
class PathConfig:
    """Singleton access to paths of important files and directories."""

    # Project root directory for all data.
    # NOTE: '..' is correct assuming CWD is directly under project root. YMMV.
    # NOTE: Use relative paths to be independent of encompassing filesystem.
    root_dir: str = ".."
