#!/bin/bash

source "$(dirname "${0}")/.venv/bin/activate"

out='.venv/tmp/performance.profile'
[ -e "${out}" ] || python -m cProfile -o "${out}" rh.py tabulize -v losers
[ $? -ne 0 ] || runsnake "${out}"
exit $?
