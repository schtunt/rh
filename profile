#!/bin/bash

source "$(dirname "${0}")/.venv/bin/activate"

echo "$*" > '.venv/tmp/performance.cmd'
out='.venv/tmp/performance.profile'
python -m cProfile -o "${out}" rh.py "$@" && runsnake "${out}"
exit $?
