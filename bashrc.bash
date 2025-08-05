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
PY_VERSION=python3.10
export PY_VERSION
export PYTHONPATH=$PYTHONPATH:"$COMA_PROJECT_ROOT_DIR"/.venv/lib/"$PY_VERSION"/site-packages

# Environment variables for launching without commands and configs.
export COMA_DEFAULT_CONFIG_DIR="$COMA_PROJECT_ROOT_DIR"/launch
export COMA_DEFAULT_COMMAND="test.launch"

# Create the launch config directory.
mkdir -p "$COMA_DEFAULT_CONFIG_DIR"

# Alias for program entry.
launch () {
  pushd "$COMA_DEFAULT_CONFIG_DIR" > /dev/null || exit
  "$PY_VERSION" "$COMA_PROJECT_ROOT_DIR"/src/main.py "$@"
  popd > /dev/null || exit
}
export -f launch

# Basic terminal auto-complete.
complete -W "
exp.1.preprocess
exp.1.compare.conceptnet.spacy
exp.1.make.prompts
exp.1.count.errors
exp.1.analysis
exp.2.make.prompts
exp.2.test.accord.loader
test.launch
test.linguistic.features
test.load.conceptnet
test.logprob.alignment
" launch

# Alias for commands launched from scripts rather than directly.
launch-llm () {
  pushd "$COMA_DEFAULT_CONFIG_DIR" > /dev/null || exit
  bash vec_inf.bash "$@"
  popd > /dev/null || exit
}
export -f launch-llm

launch-exp1-infer () {
  pushd "$COMA_DEFAULT_CONFIG_DIR" > /dev/null || exit
  bash experiment1/infer.bash "$@"
  popd > /dev/null || exit
}
export -f launch-exp1-infer

launch-exp1-infer-loop () {
  START_TIME=$(date +%s)
  for q in ALL_RELATIONS SAME_RELATION OTHER_RELATIONS SAME_SOURCE ; do
    for f in TRIPLET ACCORD ; do
      echo concept_net_query_method="$q" data_format_method="$f"
      launch-exp1-infer "$@" -- concept_net_query_method="$q" data_format_method="$f"
    done
  done
  END_TIME=$(date +%s)
  echo "Completed all runs in $((END_TIME - START_TIME)) total seconds."
}

launch-exp1-analysis-loop () {
  START_TIME=$(date +%s)
  for q in ALL_RELATIONS SAME_RELATION OTHER_RELATIONS SAME_SOURCE ; do
    for f in TRIPLET ACCORD ; do
      echo concept_net_query_method="$q" data_format_method="$f"
      launch exp.1.analysis concept_net_query_method="$q" data_format_method="$f"
    done
  done
  END_TIME=$(date +%s)
  echo "Completed all runs in $((END_TIME - START_TIME)) total seconds."
}

launch-exp2-infer () {
  pushd "$COMA_DEFAULT_CONFIG_DIR" > /dev/null || exit
  bash experiment2/infer.bash "$@"
  popd > /dev/null || exit
}
export -f launch-exp2-infer

launch-exp2-infer-loop () {
  START_TIME=$(date +%s)
  for subset in BASELINE ONE TWO THREE FOUR FIVE ; do
    echo subset="$subset"
    launch-exp2-infer "$@" -- subset="$subset"
  done
  END_TIME=$(date +%s)
  echo "Completed all runs in $((END_TIME - START_TIME)) total seconds."
}
