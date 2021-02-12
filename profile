#!/bin/bash

source "$(dirname "${0}")/.venv/bin/activate"

out='.venv/tmp/performance.profile'
[ -e "${out}" ] || python -m cProfile -o ${out} python rh.py tabulize -v losers -s ticker

runsnake "${out}"
exit $?
