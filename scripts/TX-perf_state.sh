#!/bin/sh
# This script displays the state of various configuration options on the TX1.
echo "WARNING - Must Be Run Sudo"

echo "WARNING - Use Only on TX1"

echo "Fan setting"
cat /sys/kernel/debug/tegra_fan/target_pwm

echo "Cores active"
cat /sys/devices/system/cpu/online

echo "Scaling governors (0, 1, 2, 3)"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu1/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu2/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu3/cpufreq/scaling_governor

echo "CPU available frequencies"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies

echo "CPU cycle frequencies (0, 1, 2, 3)"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq
cat /sys/devices/system/cpu/cpu1/cpufreq/scaling_min_freq
cat /sys/devices/system/cpu/cpu2/cpufreq/scaling_min_freq
cat /sys/devices/system/cpu/cpu3/cpufreq/scaling_min_freq

echo "Quiet enabled?"
cat /sys/devices/system/cpu/cpuquiet/tegra_cpuquiet/enable

echo "Throttling"
cat /proc/sys/kernel/sched_rt_runtime_us

echo "GPU available clock rates (Hz):"
cat /sys/kernel/debug/clock/gbus/possible_rates

echo "GPU clock rate (Hz):"
cat /sys/kernel/debug/clock/gbus/rate

echo "Memory available cycle rate (Hz)"
cat /sys/kernel/debug/clock/emc/possible_rates

echo "Memory cycle rate (Hz)"
cat /sys/kernel/debug/clock/emc/rate

echo "End Performance States"
