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

# Colors: http://www.tcl.tk/man/tcl8.4/TkCmd/colors.htm
idToColorMap = {0: 'light pink',
                1: 'light blue',
                2: 'LightGoldenrod2',
                3: 'light sea green',
                4: 'MediumPurple1',
                5: 'gray68',
                6: 'orange',
                7: 'gray32'}

patternColorToBgColorMap = {"light pink": "lavender blush",
                            "light blue": "azure",
                            "LightGoldenrod2": "light yellow",
                            "light sea green": "DarkSeaGreen1",
                            "MediumPurple1": "lavender",
                            "gray68": "light gray",
                            "orange": "navajo white",
                            "gray32": "gray48"}

patternColorToArrowColorMap = {"light pink": "IndianRed3",
                               "light blue": "SteelBlue2",
                               "LightGoldenrod2": "LightGoldenrod3",
                               "light sea green": "SpringGreen4",
                               "MediumPurple1": "MediumPurple1",
                               "gray68": "gray68",
                               "orange": "dark orange",
                               "gray32": "gray32"}

BUFFER_TOP = 60
BUFFER_BOTTOM = 100

BUFFER_LEFT = 100
BUFFER_RIGHT = 20

LEGEND_HEIGHT = 70
BUFFER_LEGEND = 4

USE_PATTERNS = True
LINE_WIDTH = 1

class Pattern(object):
    def __init__(self):
        self.objs = []

    def draw(self, canvas):
        for obj in self.objs:
            obj.draw(canvas)

class HorizontalLinePattern(Pattern):
    NUMLINES = 12

    def __init__(self, rect, color, numLines = None):
        Pattern.__init__(self)
        if numLines == None:
            numLines = self.NUMLINES

        p1x = rect.getP1().x+1
        p2x = rect.getP2().x-1

        ymin = min(rect.getP1().y, rect.getP2().y)
        ymax = max(rect.getP1().y, rect.getP2().y)
        h = ymax - ymin
        dy = h / float(numLines + 1)

        for i in range(1, numLines+1):
            pos = ymin + dy * i
            line = Line(Point(p1x, pos), Point(p2x, pos))
            line.setOutline(color)
            line.setWidth(LINE_WIDTH)
            self.objs.append(line)

class VerticalLinePattern(Pattern):
    NUMLINES = 23

    def __init__(self, rect, color, numLines = None):
        Pattern.__init__(self)
        if numLines == None:
            numLines = self.NUMLINES

        p1y = rect.getP1().y-1
        p2y = rect.getP2().y+1

        xmin = min(rect.getP1().x, rect.getP2().x)
        xmax = max(rect.getP1().x, rect.getP2().x)
        w = xmax - xmin
        dx = w / float(numLines + 1)

        for i in range(1, numLines+1):
            pos = xmin + dx * i
            line = Line(Point(pos, p1y), Point(pos, p2y))
            line.setOutline(color)
            line.setWidth(LINE_WIDTH)
            self.objs.append(line)

class LeftDiagonalLinePattern(Pattern):
    NUMLINES = 30

    def __init__(self, rect, color, numLines = None):
        Pattern.__init__(self)
        if numLines == None:
            numLines = self.NUMLINES

        xmin = int(min(rect.getP1().x, rect.getP2().x))
        xmax = int(max(rect.getP1().x, rect.getP2().x))
        ymin = int(min(rect.getP1().y, rect.getP2().y))
        ymax = int(max(rect.getP1().y, rect.getP2().y))

        totalDist = (ymax - ymin) + (xmax - xmin)
        ddist = totalDist / float(numLines + 1)

        ycount = int((ymax-ymin) / ddist)
        xcount = int((xmax-xmin) / ddist)

        for i in range(ycount+1):
            x1 = xmin + 1
            y1 = ymin + ddist * i

            y2 = ymax - 1
            x2 = (y2 - y1) + x1

            line = Line(Point(x1, y1), Point(x2, y2))
            line.setOutline(color)
            line.setWidth(LINE_WIDTH)
            self.objs.append(line)

        for i in range(1, xcount+1):
            x1 = xmin + ddist * i
            y1 = ymin + 1

            x2 = (ymax - y1) + x1 - 1
            y2 = ymax - 1
            if x2 > xmax:
                y2 -= x2 - xmax + 1
                x2 = xmax - 1

            line = Line(Point(x1, y1), Point(x2, y2))
            line.setOutline(color)
            line.setWidth(LINE_WIDTH)
            self.objs.append(line)

class RightDiagonalLinePattern(Pattern):
    NUMLINES = 30

    def __init__(self, rect, color, numLines = None):
        Pattern.__init__(self)
        if numLines == None:
            numLines = self.NUMLINES

        xmin = int(min(rect.getP1().x, rect.getP2().x))
        xmax = int(max(rect.getP1().x, rect.getP2().x))
        ymin = int(min(rect.getP1().y, rect.getP2().y))
        ymax = int(max(rect.getP1().y, rect.getP2().y))

        totalDist = (ymax - ymin) + (xmax - xmin)
        ddist = totalDist / float(numLines + 1)

        ycount = int((ymax-ymin) / ddist)
        xcount = int((xmax-xmin) / ddist)

        for i in range(ycount+1):
            x2 = xmax - 1
            y2 = ymin + ddist * i #ymax - ddist * i

            y1 = ymax - 1
            x1 = xmax - (ymax - y2)

            line = Line(Point(x1, y1), Point(x2, y2))
            line.setOutline(color)
            line.setWidth(LINE_WIDTH)
            self.objs.append(line)

        for i in range(1, xcount+1):
            x2 = xmax - ddist * i
            y2 = ymin + 1

            y1 = ymax - 1
            x1 = x2 - (y1 - ymin) + 1
            if x1 < xmin:
                y1 -= xmin - x1 + 1
                x1 = xmin + 1

            line = Line(Point(x1, y1), Point(x2, y2))
            line.setOutline(color)
            line.setWidth(LINE_WIDTH)
            self.objs.append(line)

idToPatternMap = {0: HorizontalLinePattern,
                  1: RightDiagonalLinePattern,
                  2: VerticalLinePattern,
                  3: LeftDiagonalLinePattern,
                  4: VerticalLinePattern,
                  5: RightDiagonalLinePattern,
                  6: LeftDiagonalLinePattern,
                  7: HorizontalLinePattern}

class PlotRect(Rectangle):
    def __init__(self, w, h):
        # Bottom left
        p1x = BUFFER_LEFT
        p1y = h - BUFFER_BOTTOM

        # Top right
        p2x = w - BUFFER_RIGHT
        p2y = BUFFER_TOP + LEGEND_HEIGHT + BUFFER_LEGEND

        Rectangle.__init__(self, Point(p1x, p1y), Point(p2x, p2y))
        self.setFill("white")

class BlockSMRect(object):
    def __init__(self, block, firstTime, totalTime, totalNumSms, w, h, color, patternType, otherThreads):
        self.build_rectangle(block, firstTime, totalTime, totalNumSms, w, h, color, patternType, otherThreads)
        self.build_label(block, firstTime, totalTime, totalNumSms, w, h, color, otherThreads)

    def build_rectangle(self, block, firstTime, totalTime, totalNumSms, w, h, color, patternType, otherThreads):
        # Height is fraction of an SM: 2048 threads/SM, with block.numThreads threads in block
        plotHeight = h - BUFFER_TOP - LEGEND_HEIGHT - BUFFER_LEGEND - BUFFER_BOTTOM
        plotBottom = h - BUFFER_BOTTOM
        smHeight = plotHeight / totalNumSms
        smBottom = plotBottom - int((block.sm) * smHeight)
        blockHeight = smHeight / 2048.0 * block.numThreads

        otherHeight = smHeight / 2048.0 * otherThreads
        blockBottom = smBottom - otherHeight
        blockTop = blockBottom - blockHeight
        
        p1x = int(float(block.start - firstTime) / totalTime * (w-BUFFER_LEFT-BUFFER_RIGHT)) + BUFFER_LEFT
        p1y = blockBottom

        p2x = int(float(block.end - firstTime) / totalTime * (w-BUFFER_LEFT-BUFFER_RIGHT)) + BUFFER_LEFT
        p2y = blockTop

        self.block = Rectangle(Point(p1x, p1y), Point(p2x, p2y))
        if patternType == None:
            self.block.setFill(color)
            self.pattern = None
        else:
            self.block.setFill(patternColorToBgColorMap[color])
            self.pattern = patternType(self.block, color)

    def build_label(self, block, firstTime, totalTime, totalNumSms, w, h, color, otherThreads):
        # Height is fraction of an SM: 2048 threads/SM, with block.numThreads threads in block
        plotHeight = h - BUFFER_TOP - LEGEND_HEIGHT - BUFFER_LEGEND - BUFFER_BOTTOM
        plotBottom = h - BUFFER_BOTTOM
        smHeight = plotHeight / totalNumSms
        smBottom = plotBottom - int((block.sm) * smHeight)
        blockHeight = smHeight / 2048.0 * block.numThreads

        otherHeight = smHeight / 2048.0 * otherThreads
        blockBottom = smBottom - otherHeight
        blockTop = blockBottom - blockHeight

        blockLeft = int(float(block.start - firstTime) / totalTime * (w-BUFFER_LEFT-BUFFER_RIGHT)) + BUFFER_LEFT
        blockRight = int(float(block.end - firstTime) / totalTime * (w-BUFFER_LEFT-BUFFER_RIGHT)) + BUFFER_LEFT

        px = (blockLeft + blockRight) / 2
        py = (blockTop + blockBottom) / 2

        kernelName = block.kernelName
        self.label = Text(Point(px, py), "%s: %s" % (kernelName, block.id))

    def draw(self, canvas):
        self.block.draw(canvas)
        if self.pattern != None:
            self.pattern.draw(canvas)
        self.label.draw(canvas)

class KernelReleaseMarker(Line):
    def __init__(self, kernel, firstTime, totalTime, totalNumSms, w, h, color, patternType, idx):
        releaseTime = kernel.releaseTime

        px = int(float(releaseTime - firstTime) / totalTime * (w-BUFFER_LEFT-BUFFER_RIGHT)) + BUFFER_LEFT
        p1y = h - BUFFER_BOTTOM + 20 + idx * 20
        p2y = p1y + 20

        Line.__init__(self, Point(px, p1y), Point(px, p2y))
        self.setArrow("first")
        self.setWidth(2)
        if patternType == None:
            self.setFill(color)
        else:
            self.setFill(patternColorToArrowColorMap[color])

class Title(object):
    def __init__(self, w, h, name):
        self.build_title(w, h, name)

    def build_title(self, w, h, name):
        px = w / 2
        py = int(BUFFER_TOP * 0.75)

        self.title = Text(Point(px, py), name)

    def draw(self, canvas):
        self.title.draw(canvas)

class LegendBox(object):
    def __init__(self, posx, posy, w, h, i):
        self.rect = Rectangle(Point(posx - w/2, posy - h/2), Point(posx + w/2, posy + h/2))

        if USE_PATTERNS:
            color = idToColorMap[i]
            self.rect.setFill(patternColorToBgColorMap[color])

            patternType = idToPatternMap[i]
            if patternType == HorizontalLinePattern:
                self.pattern = patternType(self.rect, color, 2)
            elif patternType == VerticalLinePattern:
                self.pattern = patternType(self.rect, color, 2)
            elif patternType == LeftDiagonalLinePattern:
                self.pattern = patternType(self.rect, color, 3)
            elif patternType == RightDiagonalLinePattern:
                self.pattern = patternType(self.rect, color, 3)
        else:
            color = patternColorToArrowColorMap[idToColorMap[i]]
            self.rect.setFill(color)
            self.pattern = None

    def draw(self, canvas):
        self.rect.draw(canvas)
        if self.pattern != None:
            self.pattern.draw(canvas)

class Legend(object):
    def __init__(self, w, h, benchmark):
        self.build_rectangle(w, h)
        self.build_labels(w, h, benchmark)

    def build_rectangle(self, w, h):
        # Top left
        p1x = BUFFER_LEFT
        p1y = BUFFER_TOP

        # Bottom right
        p2x = w - BUFFER_RIGHT
        p2y = p1y + LEGEND_HEIGHT

        self.outline = Rectangle(Point(p1x, p1y), Point(p2x, p2y))
        self.outline.setFill("white")
        self.outline.setOutline("black")

    def build_labels(self, w, h, benchmark):
        self.boxes = []
        self.labels = []
        for i in range(len(benchmark.kernels)):
            kernel = benchmark.kernels[i]
            self.build_label(w, h, kernel, i, len(benchmark.kernels))

    def build_label(self, w, h, kernel, i, n):
        boxSize = 12
        numPerCol = math.ceil(n / 2.0)

        if i < numPerCol:
            # Left half
            row = i

            left = BUFFER_LEFT
            right = left + (w - BUFFER_LEFT - BUFFER_RIGHT) / 2
            midx = (left + right) / 2
        else:
            # Right half
            row = i - numPerCol

            left = BUFFER_LEFT + (w - BUFFER_LEFT - BUFFER_RIGHT) / 2
            right = w - BUFFER_RIGHT
            midx = (left + right) / 2

        top = BUFFER_TOP + (LEGEND_HEIGHT / numPerCol) * row
        bottom = top + (LEGEND_HEIGHT / numPerCol)
        midy = (top + bottom) / 2

        # Build the box
        box = LegendBox(left + 20, midy, boxSize, boxSize, i)
        self.boxes.append(box)

        # Build the label
        s = kernel.label
        label = Text(Point(left + 100, midy), "Stream %d (%s)" % (i+1, s)) # TODO: generalize
        self.labels.append(label)

    def draw(self, canvas):
        self.outline.draw(canvas)
        for box in self.boxes:
            box.draw(canvas)
        for label in self.labels:
            label.draw(canvas)

class XAxis(object):
    def __init__(self, firstTime, totalTime, w, h):
        self.build_axis(w, h)

        self.calculate_tick_time(totalTime)
        self.build_tick_marks(totalTime, w, h)
        self.build_labels(totalTime, w, h)

    def build_axis(self, w, h):
        # Draw a thin black horizontal line
        p1x = BUFFER_LEFT
        p2x = w - BUFFER_RIGHT

        py = h - BUFFER_BOTTOM

        self.axis = Line(Point(p1x, py), Point(p2x, py))
        self.axis.setFill("black")

    def calculate_tick_time(self, totalTime):
        if totalTime <= 2.0:
            self.tick_time = 0.1
        elif totalTime <= 4.0:
            self.tick_time = 0.2
        else:
            self.tick_time = 0.5

    def build_tick_marks(self, totalTime, w, h):
        # Put a tick every 0.1 seconds
        plotWidth = w - BUFFER_LEFT - BUFFER_RIGHT
        numTicks = int(math.ceil(totalTime / self.tick_time))
        self.ticks = []
        for i in range(1, numTicks):
            # Top of plot area
            px = BUFFER_LEFT + (i * self.tick_time / totalTime) * plotWidth
            p1y = BUFFER_TOP + LEGEND_HEIGHT + BUFFER_LEGEND + 1
            p2y = BUFFER_TOP + LEGEND_HEIGHT + BUFFER_LEGEND + 5

            tick = Line(Point(px, p1y), Point(px, p2y))
            tick.setFill("black")
            self.ticks.append(tick)

            # Bottom of plot area
            px = BUFFER_LEFT + (i * self.tick_time / totalTime) * plotWidth
            p1y = h - BUFFER_BOTTOM - 0
            p2y = h - BUFFER_BOTTOM - 4

            tick = Line(Point(px, p1y), Point(px, p2y))
            tick.setFill("black")
            self.ticks.append(tick)

    def build_labels(self, totalTime, w, h):
        # Put a label every tick mark
        plotWidth = w - BUFFER_LEFT - BUFFER_RIGHT
        numTicks = int(math.ceil(totalTime / self.tick_time))
        self.labels = []
        for i in range(1, numTicks):
            px = BUFFER_LEFT + (i * self.tick_time / totalTime) * plotWidth
            py = h - int(BUFFER_BOTTOM * 0.9)

            label = Text(Point(px, py), "%.1f" % (i * self.tick_time))
            label.setSize(10)
            self.labels.append(label)

        # Give the axis a label
        px = w / 2
        py = h - (BUFFER_BOTTOM * 0.6)
        label = Text(Point(px, py), "Time (seconds)")
        self.labels.append(label)

    def draw(self, canvas):
        self.axis.draw(canvas)
        for tick in self.ticks:
            tick.draw(canvas)
        for label in self.labels:
            label.draw(canvas)

class YAxis(Rectangle):
    def __init__(self, totalNumSms, firstTime, totalTime, w, h):
        self.build_axis(w, h)
        self.build_grid_lines(totalNumSms, w, h)
        self.build_labels(totalNumSms, w, h)

    def build_axis(self, w, h):
        # Draw a thin black vertical line
        px = BUFFER_LEFT

        p1y = h - BUFFER_BOTTOM
        p2y = BUFFER_TOP + LEGEND_HEIGHT + BUFFER_LEGEND

        self.axis = Line(Point(px, p1y), Point(px, p2y))
        self.axis.setFill("black")

    def build_grid_lines(self, totalNumSms, w, h):
        # Put a horizontal line between each SM
        plotHeight = h - BUFFER_TOP - LEGEND_HEIGHT - BUFFER_LEGEND - BUFFER_BOTTOM
        plotBottom = h - BUFFER_BOTTOM
        smHeight = plotHeight / totalNumSms
        self.gridlines = []
        for i in range(1, totalNumSms):
            py = plotBottom - i * smHeight

            p1x = BUFFER_LEFT
            p2x = w - BUFFER_RIGHT

            line = Line(Point(p1x, py), Point(p2x, py))
            line.setFill("black")
            self.gridlines.append(line)

    def build_labels(self, totalNumSms, w, h):
        plotHeight = h - BUFFER_TOP - LEGEND_HEIGHT - BUFFER_LEGEND - BUFFER_BOTTOM
        plotBottom = h - BUFFER_BOTTOM
        smHeight = plotHeight / totalNumSms
        self.labels = []
        for i in range(totalNumSms):
            px = int(BUFFER_LEFT * 0.75)
            py = plotBottom - i * smHeight - int(0.5 * smHeight)

            label = Text(Point(px, py), "SM %d" % i)
            label.setSize(10)
            self.labels.append(label)

    def draw(self, canvas):
        self.axis.draw(canvas)
        for line in self.gridlines:
            line.draw(canvas)
        for label in self.labels:
            label.draw(canvas)

class ResizingCanvasFrame(CanvasFrame):
    def __init__(self, parent, width, height, redraw_callback):
        CanvasFrame.__init__(self, parent, width, height)
        self.parent = parent
        self.parent.resizable(True, True)

        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self.on_resize)
        self.canvas.config(highlightthickness=0)
        self.redrawCallback = redraw_callback

    def on_resize(self, event):
        self.redrawCallback(event.width, event.height)

    def clear_canvas(self):
        self.canvas.delete("all")

class BlockSMDisplay():
    INIT_WIDTH = 800
    INIT_HEIGHT = 600
    
    def __init__(self, win, benchmark):
        self.setup(win, self.INIT_WIDTH, self.INIT_HEIGHT, benchmark)
        self.draw_benchmark()

    def setup(self, win, width, height, benchmark):
        self.width = width
        self.height = height
        self.firstTime = 0.0
        self.totalTime = (benchmark.get_end() - self.firstTime) * 1.1

        # Create a canvas
        self.canvas = ResizingCanvasFrame(win, self.width, self.height, self.redraw)
        self.canvas.setBackground("light gray")

        self.benchmark = benchmark

        if len(benchmark.kernels) > 0:
            self.numSms = self.benchmark.kernels[0].maxResidentThreads / 2048
            self.name = self.benchmark.kernels[0].scenarioName

    def redraw(self, width, height):
        self.canvas.clear_canvas()

        self.width = width
        self.height = height
        self.draw_benchmark()

    def draw_benchmark(self):
        if len(self.benchmark.kernels) == 0: return
        
        # Draw the plot area
        self.draw_plot_area()

        # Draw each kernel
        smBase = [[] for j in range(self.numSms)]
        releaseDict = {}
        for i in range(len(self.benchmark.kernels)):
            color = idToColorMap[i] if USE_PATTERNS else patternColorToArrowColorMap[idToColorMap[i]]
            patternType = idToPatternMap[i] if USE_PATTERNS else None
            self.draw_kernel(self.benchmark.kernels[i], color, patternType, i, smBase, releaseDict)

        # Draw the title, legend, and axes
        self.draw_title()
        self.draw_legend()
        self.draw_axes()

    def draw_plot_area(self):
        pr = PlotRect(self.width, self.height)
        pr.draw(self.canvas)

    def draw_kernel(self, kernel, color, patternType, i, smBase, releaseDict):
        # Draw each block of the kernel
        for block in kernel.blocks:
            # Calculate competing threadcount
            otherThreads = 0
            for interval in smBase[block.sm]:
                if interval[0] < block.end and interval[1] > block.start:
                    otherThreads = max(otherThreads, interval[2])

            br = BlockSMRect(block, self.firstTime, self.totalTime, self.numSms,
                             self.width, self.height, color, patternType, otherThreads)

            br.draw(self.canvas)

            smBase[block.sm].append((block.start, block.end, block.numThreads))

        # Draw a marker for the kernel release time
        releaseIdx = releaseDict.get(kernel.releaseTime, 0)
        releaseDict[kernel.releaseTime] = releaseIdx + 1
        krm = KernelReleaseMarker(kernel, self.firstTime, self.totalTime,
                                  self.numSms, self.width, self.height, color, patternType, releaseIdx)
        krm.draw(self.canvas)

    def draw_title(self):
        title = Title(self.width, self.height, self.name)
        title.draw(self.canvas)

    def draw_legend(self):
        legend = Legend(self.width, self.height, self.benchmark)
        legend.draw(self.canvas)

    def draw_axes(self):
        xaxis = XAxis(self.firstTime, self.totalTime, self.width, self.height)
        xaxis.draw(self.canvas)

        yaxis = YAxis(self.numSms, self.firstTime, self.totalTime, self.width, self.height)
        yaxis.draw(self.canvas)

###################################################
# Data                                            #
###################################################

class Block(object):
    def __init__(self, startTime, endTime, numThreads, smId, threadId, blockId, kernel):
        self.start = startTime
        self.end = endTime
        self.numThreads = numThreads
        self.sm = smId
        self.thread = threadId
        self.id = blockId
        self.kernelName = kernel.kernelName
        self.kernel = kernel

class Kernel(object):
    def __init__(self, benchmark):
        self.parse_benchmark(benchmark)

    def parse_benchmark(self, benchmark):        
        self.blocks = []

        self.scenarioName = benchmark["scenario_name"]
        self.label = benchmark["label"] # string
        self.releaseTime = benchmark["release_time"] # float
        self.blockCount = benchmark["block_count"] # int
        self.threadCount = benchmark["thread_count"] # int
        self.maxResidentThreads = benchmark["max_resident_threads"] # int

        self.kernelName = self.label

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
                          self.blockSms[-1], benchmark["TID"], i, self)
            self.blocks.append(block)
        
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
