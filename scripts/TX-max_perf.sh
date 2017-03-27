#!/bin/sh
# This script manually sets the TX1's GPU and memory clock rates to a high
# value, disables frequency scaling, puts the machine in "performance" mode,
# and turns on the fan.
echo "WARNING - Must Be Run Sudo"

echo "WARNING - Use Only on TX1"

echo "Turn on fan for safety"
echo 255 > /sys/kernel/debug/tegra_fan/target_pwm
echo "Fan setting"
cat /sys/kernel/debug/tegra_fan/target_pwm

echo "Cores active"
cat /sys/devices/system/cpu/online

echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
echo performance > /sys/devices/system/cpu/cpu1/cpufreq/scaling_governor
echo performance > /sys/devices/system/cpu/cpu2/cpufreq/scaling_governor
echo performance > /sys/devices/system/cpu/cpu3/cpufreq/scaling_governor

echo "Scaling governors (0, 1, 2, 3)"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu1/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu2/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu3/cpufreq/scaling_governor

echo "CPU available frequencies"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies

cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq > /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq
cat /sys/devices/system/cpu/cpu1/cpufreq/scaling_max_freq > /sys/devices/system/cpu/cpu1/cpufreq/scaling_min_freq
cat /sys/devices/system/cpu/cpu2/cpufreq/scaling_max_freq > /sys/devices/system/cpu/cpu2/cpufreq/scaling_min_freq
cat /sys/devices/system/cpu/cpu3/cpufreq/scaling_max_freq > /sys/devices/system/cpu/cpu3/cpufreq/scaling_min_freq

echo "CPU minimum cycle frequencies (0, 1, 2, 3)"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq
cat /sys/devices/system/cpu/cpu1/cpufreq/scaling_min_freq
cat /sys/devices/system/cpu/cpu2/cpufreq/scaling_min_freq
cat /sys/devices/system/cpu/cpu3/cpufreq/scaling_min_freq

echo 0 > /sys/devices/system/cpu/cpuquiet/tegra_cpuquiet/enable
echo "Quiet enabled?"
cat /sys/devices/system/cpu/cpuquiet/tegra_cpuquiet/enable

echo -1 >/proc/sys/kernel/sched_rt_runtime_us
echo "Throttling"
cat /proc/sys/kernel/sched_rt_runtime_us

echo "GPU available clock rates (Hz):"
cat /sys/kernel/debug/clock/gbus/possible_rates
echo 844800000 > /sys/kernel/debug/clock/override.gbus/rate
echo 1 > /sys/kernel/debug/clock/override.gbus/state
echo "GPU clock rate (Hz):"
cat /sys/kernel/debug/clock/gbus/rate

echo "Memory available cycle rate (Hz)"
cat /sys/kernel/debug/clock/emc/possible_rates
echo 1331200000 > /sys/kernel/debug/clock/override.emc/rate
echo 1 > /sys/kernel/debug/clock/override.emc/state
echo "Memory cycle rate (Hz)"
cat /sys/kernel/debug/clock/emc/rate

echo "Max Performance Settings Done"
