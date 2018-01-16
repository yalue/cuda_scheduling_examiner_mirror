#!/bin/bash
# This script runs VisionWorks demos in 4 different scenarios with multiple
# instances of each demo. These scenarios are multi-thread (MT), multi-thread
# with MPS (MT MPS), multi-process (MP), multi-process with MPS (MP MPS).
#
# nvprof log files are produced into ./results/nvvp/ directory

mkdir ./results/nvvp

titles=("Video Stabilizer" "Feature Tracker" "Hough Transform" "Motion Estimation" "Stereo Matching")
options=("vs" "ft" "ht" "me" "sm")

for i in `seq 0 1 4`;
do
    title=${titles[$i]}
    option=${options[$i]} # option name for the demo to run
    for j in 1 2 4 #8 16
    do
        echo $j "$title instance(s)"

        # MP
        ./scripts/stopMPS.sh
        nvprof --profile-all-processes -o results/nvvp/$option"_x"$j"_mp_%p.nvvp" &
        nvprof_pid=$!
        python ./scripts/visionworks_generator.py --$option $j --name "$title" -p | ./bin/runner -
        kill $nvprof_pid
        wait

        # MT
        nvprof --profile-all-processes -o results/nvvp/$option"_x"$j"_mt_%p.nvvp" &
        nvprof_pid=$!
        python ./scripts/visionworks_generator.py --$option $j --name "$title" | ./bin/runner -
        kill $nvprof_pid
        wait

        # MP MPS
        nvprof --profile-all-processes -o results/nvvp/$option"_x"$j"_mp_mps_%p.nvvp" &
        nvprof_pid=$!
        ./scripts/startMPS.sh
        python ./scripts/visionworks_generator.py --$option $j --name "$title" -p -m | ./bin/runner -
        kill $nvprof_pid
        wait

        # MT MPS
        nvprof --profile-all-processes -o results/nvvp/$option"_x"$j"_mt_mps_%p.nvvp" &
        nvprof_pid=$!
        ./scripts/startMPS.sh
        python ./scripts/visionworks_generator.py --$option $j --name "$title" -m | ./bin/runner -
        kill $nvprof_pid
        wait
    done
done
