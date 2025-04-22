from types import ModuleType
from typing import Union
import importlib
import pkgutil

from coma import command


ModuleName = str


def import_submodules(
    package: Union[ModuleName, ModuleType], recursive: bool = True
) -> dict[ModuleName, ModuleType]:
    """Import all submodules, optionally recursively."""
    # Credit: https://stackoverflow.com/questions/3365740/how-to-import-all-submodules/25083161#25083161
    if isinstance(package, ModuleName):
        package = importlib.import_module(package)
    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + "." + name
        try:
            results[full_name] = importlib.import_module(full_name)
        except ModuleNotFoundError:
            continue
        if recursive and is_pkg:
            results.update(import_submodules(full_name))
    return results


@command(name="test.launch")
def test_launch():
    """Make sure the environment is correctly set up for launching the program."""
    print("Successfully launched.")


__all__ = ["test_launch"] + list(import_submodules(__name__).keys())
