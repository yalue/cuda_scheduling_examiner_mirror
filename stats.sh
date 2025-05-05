#!/bin/bash
cat $1 | jq "[.times[].execute_times | select(. != null) | (.[1] - .[0]) * 1000] | max, add/length, min" | xargs printf "max: %.2f\navg: %.2f\nmin: %.2f\n"
