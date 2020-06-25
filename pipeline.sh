#!/bin/bash

if [[ -n "$@" ]]; then
    for arg in "$@" # Iterate over arguments
    do
        /home/anaconda3/bin/python build_graph.py "$arg" &&
        /home/anaconda3/bin/python project_graph.py "$arg" &&
        /home/anaconda3/bin/python compute_knc.py "$arg" &&
        /home/anaconda3/bin/python analyze_knc.py "$arg"
    done
else
    echo "[Info] Please specify run config files as arguments"
fi
