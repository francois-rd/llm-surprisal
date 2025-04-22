#!/bin/bash

# Find the project root directory assuming this script file lives directly inside it.
COMA_PROJECT_ROOT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
export COMA_PROJECT_ROOT_DIR

# Add main and plugin code to PYTHONPATH.
if [ -z "$PYTHONPATH" ]
then
  export PYTHONPATH="$COMA_PROJECT_ROOT_DIR"/src
else
  export PYTHONPATH=$PYTHONPATH:"$COMA_PROJECT_ROOT_DIR"/src
fi

# Library path (needed only for Pycharm because of seeming bug).
# NOTE: Change your python version if needed.
export PYTHONPATH=$PYTHONPATH:"$COMA_PROJECT_ROOT_DIR"/.venv/lib/python3.9/site-packages

# Environment variables for launching without commands and configs.
export COMA_DEFAULT_CONFIG_DIR="$COMA_PROJECT_ROOT_DIR"/launch
export COMA_DEFAULT_COMMAND="test.launch"

# Create the launch config directory.
mkdir -p "$COMA_DEFAULT_CONFIG_DIR"

# Alias for program entry.
launch () {
  pushd "$COMA_DEFAULT_CONFIG_DIR" > /dev/null || exit
  python "$COMA_PROJECT_ROOT_DIR"/src/main.py "$@"
  popd > /dev/null || exit
}
export -f launch

# Basic terminal auto-complete.
complete -W "
test.launch
" launch
