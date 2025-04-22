import logging
import os

from coma import InstanceKeys, InvocationData, WakeException, wake
import coma

from ..io import logging as log
from .commands import test_launch  # This also imports all commands in all submodules.


# ===== Parser hooks =====

overwrite_hook = coma.add_argument_factory(
    "--overwrite",
    nargs=1,
    default="",
    type=str,
    choices=[
        InstanceKeys.BASE.lower(),
        InstanceKeys.FILE.lower(),
        InstanceKeys.OVERRIDE.lower(),
    ],
    metavar="INSTANCE_KEY",
    help="if given, overwrite *all* config files with the given instance variant",
)
create_hook = coma.add_argument_factory(
    "--create",
    action="store_true",
    help="exit during post-config",
)
dry_run_hook = coma.add_argument_factory(
    "--dry-run",
    action="store_true",
    help="exit during pre-run",
)
logging_level_hook = coma.add_argument_factory(
    "--log-level",
    nargs=1,
    default="info",
    type=str,
    choices=["debug", "info", "warning", "error", "critical"],
    metavar="LEVEL",
    help="set the default global log level",
)


# ===== Invocation hooks =====


def pre_config_hook(data: InvocationData):
    """This pre-config hook sets the global default logging level."""
    log.DEFAULT_LEVEL = getattr(logging, data.known_args.log_level.upper())


def config_hook(data: InvocationData):
    """This config hook can overwrite config files if the --overwrite flag is given."""
    coma.config_hook.default_factory(
        overwrite=bool(data.known_args.overwrite),
        write_instance_key=data.known_args.overwrite.upper() or None,
    )(data)


def post_config_hook(data: InvocationData):
    """This post-config hook exist early. Useful for creating config files."""
    if data.known_args.create:
        print("Config files created. Quitting.")
        quit(0)


def pre_run_hook(data: InvocationData):
    """This pre-run hook exists early. Useful for debugging init hooks."""
    if data.known_args.dry_run:
        print("Dry run.")
        quit()


def wake_with_hooks(cli_args=None):
    """Wake from coma with application-specific non-default hooks and configs."""
    wake(
        test_launch,  # This is just so the module import isn't unused.
        cli_args=cli_args,
        parser_hook=(
            coma.DEFAULT,
            logging_level_hook,
            overwrite_hook,
            create_hook,
            dry_run_hook,
        ),
        pre_config_hook=pre_config_hook,
        config_hook=config_hook,
        post_config_hook=post_config_hook,
        pre_run_hook=pre_run_hook,
    )


def launch():
    """Launch program with default command fall back."""
    try:
        wake_with_hooks()
    except WakeException as e:
        if any("command line" in arg for arg in e.args):
            os.chdir(os.environ["COMA_DEFAULT_CONFIG_DIR"])
            wake_with_hooks(cli_args=[os.environ["COMA_DEFAULT_COMMAND"]])
        else:
            raise
