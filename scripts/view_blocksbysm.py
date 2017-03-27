# This script reads all JSON result files and uses matplotlib to display a
# timeline indicating when blocks and threads from multiple jobs were run on
# GPU, including which SM they ran on. For this to work, all result filenames
# must end in .json.
#
# Usage: python view_blocksbysm.py [results directory (default: ./results)]
import glob
import json
import math
import sys

from graphics import *

###################################################
# Drawing                                         #
###################################################

idToColorMap = {0: 'blue',
                1: 'dark green',
                2: 'red',
                3: 'cyan',
                4: 'magenta',
                5: 'yellow',
                6: 'orange',
                7: 'black'}

BUFFER_TOP = 60
BUFFER_BOTTOM = 100

BUFFER_LEFT = 100

class BlockSMRect(Rectangle):    
    def __init__(self, block, firstTime, totalTime, totalNumSms, w, h, color, otherThreads):
        # Height is fraction of an SM: 2048 threads/SM, with block.numThreads threads in block
        smHeight = (h - BUFFER_TOP - BUFFER_BOTTOM) / totalNumSms
        smBottom = h - int((block.sm) * smHeight + BUFFER_BOTTOM) # h is the bottom of the window
        blockHeight = smHeight / 2048.0 * block.numThreads

        otherHeight = smHeight / 2048.0 * otherThreads
        blockBottom = smBottom - otherHeight
        blockTop = blockBottom - blockHeight
        
        p1x = int(float(block.start - firstTime) / totalTime * (w-2*BUFFER_LEFT)) + BUFFER_LEFT
        p1y = blockBottom

        p2x = int(float(block.end - firstTime) / totalTime * (w-2*BUFFER_LEFT)) + BUFFER_LEFT
        p2y = blockTop

        Rectangle.__init__(self, Point(p1x, p1y), Point(p2x, p2y))
        self.setFill(color)

class KernelReleaseRect(Rectangle):    
    def __init__(self, kernel, firstTime, totalTime, totalNumSms, w, h, color, idx):
        releaseTime = kernel.releaseTime

        p1x = int(float(releaseTime - firstTime) / totalTime * (w-2*BUFFER_LEFT)) + BUFFER_LEFT
        p1y = h - BUFFER_BOTTOM + 1 + idx * 15

        p2x = p1x + 4
        p2y = p1y + 15

        Rectangle.__init__(self, Point(p1x, p1y), Point(p2x, p2y))
        self.setFill(color)

class XAxis(Rectangle):
    def __init__(self, firstTime, totalTime, w, h):
        # Draw a thin black horizontal line
        p1x = BUFFER_LEFT
        p1y = h - BUFFER_BOTTOM

        p2x = w - BUFFER_LEFT + 20 # go a little past the right edge
        p2y = p1y - 1

        Rectangle.__init__(self, Point(p1x, p1y), Point(p2x, p2y))
        self.setFill("black")

class YAxis(Rectangle):
    def __init__(self, firstTime, totalTime, w, h):
        # Draw a thin black vertical line
        p1x = BUFFER_LEFT
        p1y = h - BUFFER_BOTTOM

        p2x = p1x - 1
        p2y = BUFFER_TOP - 20 # go a little above the top edge

        Rectangle.__init__(self, Point(p1x, p1y), Point(p2x, p2y))
        self.setFill("black")

class BlockSMDisplay():
    
    WIDTH = 800
    HEIGHT = 600
    
    def __init__(self, win, benchmark):
        self.setup(win, self.WIDTH, self.HEIGHT, benchmark)
        self.draw_benchmark()

    def setup(self, win, width, height, benchmark):
        self.width = width
        self.height = height
        self.firstTime = 0.0
        self.totalTime = benchmark.get_end() - self.firstTime

        # Create a canvas
        self.canvas = CanvasFrame(win, self.width, self.height)
        self.canvas.setBackground('light gray')

        self.benchmark = benchmark

        print "Start time:", self.firstTime
        print "End time:", self.totalTime + self.firstTime
        print "total time:", self.totalTime

    def draw_benchmark(self):
        if len(self.benchmark.kernels) == 0: return
        
        # Draw each kernel
        numSms = self.benchmark.kernels[0].maxResidentThreads / 2048
        smBase = [[] for j in range(numSms)]
        for i in range(len(self.benchmark.kernels)):
            color = idToColorMap[i]
            self.draw_kernel(self.benchmark.kernels[i], color, i, smBase)

        # Draw the axes
        self.draw_axes()

    def draw_kernel(self, kernel, color, i, smBase):
        # Draw each block of the kernel
        numSms = kernel.maxResidentThreads / 2048
        for block in kernel.blocks:
            # Calculate competing threadcount
            otherThreads = 0
            for interval in smBase[block.sm]:
                if interval[0] < block.end and interval[1] > block.start:
                    otherThreads += interval[2]

            br = BlockSMRect(block, self.firstTime, self.totalTime, numSms,
                             self.width, self.height, color, otherThreads)

            br.draw(self.canvas)

            smBase[block.sm].append((block.start, block.end, block.numThreads))

        # Draw a line for the kernel start
        krr = KernelReleaseRect(kernel, self.firstTime, self.totalTime,
                                numSms, self.width, self.height, color, i)
        krr.draw(self.canvas)

    def draw_axes(self):
        xaxis = XAxis(self.firstTime, self.totalTime, self.width, self.height)
        xaxis.draw(self.canvas)

        yaxis = YAxis(self.firstTime, self.totalTime, self.width, self.height)
        yaxis.draw(self.canvas)

###################################################
# Data                                            #
###################################################

class Block(object):
    def __init__(self, startTime, endTime, numThreads, smId, threadId):
        self.start = startTime
        self.end = endTime
        self.numThreads = numThreads
        self.sm = smId
        self.thread = threadId

class Kernel(object):
    def __init__(self, benchmark):
        self.parse_benchmark(benchmark)

    def parse_benchmark(self, benchmark):        
        self.blocks = []

        self.label = benchmark["label"] # string
        self.releaseTime = benchmark["release_time"] # float
        self.blockCount = benchmark["block_count"] # int
        self.threadCount = benchmark["thread_count"] # int
        self.maxResidentThreads = benchmark["max_resident_threads"] # int

        times = benchmark["times"][1]
        self.kernelStart = times["kernel_times"][0]
        self.kernelEnd = times["kernel_times"][1]

        self.blockStarts = []
        self.blockEnds = []
        self.blockSms = []
        for i in range(self.blockCount):
            self.blockStarts.append(times["block_times"][i*2])
            self.blockEnds.append(times["block_times"][i*2+1])
            self.blockSms.append(times["block_smids"][i])
            block = Block(self.blockStarts[-1], self.blockEnds[-1], self.threadCount,
                          self.blockSms[-1], benchmark["TID"])
            self.blocks.append(block)

        print "Parsed %d blocks" % len(self.blocks)
        
    def get_start(self):
        start = None
        for block in self.blocks:
            if start == None or block.start < start:
                start = block.start

        return start

    def get_end(self):
        end = None
        for block in self.blocks:
            if end == None or block.end > end:
                end = block.end

        return end

class Benchmark(object):
    def __init__(self, name, benchmarks):
        self.name = name
        self.parse_benchmark(benchmarks)

    def parse_benchmark(self, benchmarks):
        self.kernels = []
        for benchmark in benchmarks:
            self.kernels.append(Kernel(benchmark))

    def get_start(self):
        return min([k.get_start() for k in self.kernels])

    def get_end(self):
        return max([k.get_end() for k in self.kernels])

def get_block_intervals(name, benchmarks):
    return Benchmark(name, benchmarks)

def plot_scenario(benchmarks, name):
    """Takes a list of parsed benchmark results and a scenario name and
    generates a plot showing the timeline of benchmark behaviors for the
    specific scenario."""
    win = Window("Block Execution by SM")
    graph = BlockSMDisplay(win, get_block_intervals(name, benchmarks))
    win.mainloop()
    
def show_plots(filenames):
    """Takes a list of filenames, and generates one plot per scenario found in
    the files."""
    # Parse the files
    parsed_files = []
    for name in filenames:
        with open(name) as f:
            parsed_files.append(json.loads(f.read()))
            
    # Group the files by scenario
    scenarios = {}
    for benchmark in parsed_files:
        scenario = benchmark["scenario_name"]
        if not scenario in scenarios:
            scenarios[scenario] = []
        scenarios[scenario].append(benchmark)

    # Plot the scenarios
    for scenario in scenarios:
        plot_scenario(scenarios[scenario], scenario)

if __name__ == "__main__":
    
    base_directory = "./results"
    if len(sys.argv) > 2:
        print "Usage: python %s [directory containing results (./results)]" % (
            sys.argv[0])
        exit(1)
    if len(sys.argv) == 2:
        base_directory = sys.argv[1]
    filenames = glob.glob(base_directory + "/*.json")
    show_plots(filenames)
