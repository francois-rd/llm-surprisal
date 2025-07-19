#!/bin/bash

while getopts ":i:m:r:d" opt; do
  case ${opt} in
    i) implementation="${OPTARG}" ;;
    m) llm="${OPTARG}" ;;
    :)
      echo "Option -${OPTARG} requires an argument."
      exit 1
      ;;
    ?)
      echo "Invalid option: -${OPTARG}."
      echo "If this is an option to pass to Coma, use '--' to delineate."
      exit 1
      ;;
  esac
done

shift "$(( OPTIND - 1 ))"

if [ -z "$implementation" ]
then
  echo "Missing LLM implementation. Use the '-i' flag."
  echo "If the flag was given, make sure it appears before any non-flag arguments."
  exit 1
fi
if [ -z "$llm" ]
then
  echo "Missing LLM nickname. Use the '-m' flag."
  echo "If the flag was given, make sure it appears before any non-flag arguments."
  exit 1
fi

# Remove any lone "--" from $@
for arg
do
  shift
  [ "$arg" = "--" ] && continue
  set -- "$@" "$arg"
done

exp_dir="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
impl_lower="$(echo "$implementation" | tr '[:upper:]' '[:lower:]')"

launch exp.1.infer llm="$llm" implementation="$implementation" "$@" \
  --"$impl_lower"-path "$exp_dir"/"$llm".yaml
