#!/bin/bash
if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Print the min, mean, and max kernel execution times from a results log file."
    echo "(Useful for sanity checks on very large log files. Requires 'jq'; apt install jq.)"
    echo "Usage: $0 <results/log_file.json>"
    exit 1
fi
if [ ! -f "$1" ]; then
    echo "File '$1' does not exist."
    echo "Usage: $0 <results/log_file.json>"
    exit 1
fi
cat $1 | jq "[.times[].execute_times | select(. != null) | (.[1] - .[0]) * 1000] | max, add/length, min" | xargs printf "max: %.2f ms\navg: %.2f ms\nmin: %.2f ms\n"
