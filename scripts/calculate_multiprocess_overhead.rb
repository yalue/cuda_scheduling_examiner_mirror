require 'json'

# Takes an output JSON filename and returns the total kernel time in the file.
def get_kernel_time(filename)
  to_return = 0.0
  data = JSON.parse(File.open(filename, "rb") {|f| f.read})
  data["times"].each do |t|
    next if !t.include?("kernel_times")
    values = t["kernel_times"]
    to_return += values[1] - values[0]
  end
  to_return
end

def get_time_ratio(stdout_text)
  v = 0.0
  if stdout_text =~ /is approximately (.*?) CPU seconds./
    v = $1.to_f
  end
  v
end

# Runs a single instance of the benchmark and returns the overhead of
# multiprocess vs. multithread.
def run_single_instance()
  `rm results/*.json`
  time_ratio = nil
  while true
    if time_ratio != nil
      puts "Retrying run to get better GPU globaltimer rate."
      puts "Last rate was #{time_ratio.to_s}"
    end
    # Throw away runs in which either multiprocesses or threads don't have a
    # reasonable ratio of GPU time to real time from the globaltimer register.
    stdout_process = `./bin/runner configs/multiprocess_two_randomwalk.json`
    time_ratio = get_time_ratio(stdout_process)
    next if (time_ratio <= 0.99) || (time_ratio >= 1.01)
    stdout_thread = `./bin/runner configs/multithread_two_randomwalk.json`
    time_ratio = get_time_ratio(stdout_thread)
    next if (time_ratio <= 0.99) || (time_ratio >= 1.01)
    break
  end
  # Now, process all result files to determine the process vs. thread time.
  process_time = 0.0
  thread_time = 0.0
  files = Dir["results/*.json"]
  files.each do |f|
    kernel_time = get_kernel_time(f)
    if f =~ /multiprocess_/
      process_time += kernel_time
    elsif f =~ /multithread_/
      thread_time += kernel_time
    end
  end
  process_time / thread_time
end

iterations = 5
ratios = []
iterations.times do |t|
  ratio = run_single_instance()
  puts "Instance %d had a ratio of %f" % [t + 1, ratio]
  ratios << ratio
end
ratios.sort!
puts "Min ratio: #{ratios[0].to_f}"
puts "Max ratio: #{ratios[-1].to_f}"
puts "Median ratio: #{ratios[ratios.size / 2].to_f}"
puts "Mean ratio: #{ratios.inject(:+) / ratios.size.to_f}"
