Table of Contents
==============
1. [Overview](#overview)
2. [Install](#install)
3. [Evaluate](#evaluate)

Overview
========

Due to the necessity of using a GPU with only our benchmarks running on it, we
have not prepared a VM image.  Rather, our code is available publicly on
[GitHub:yalue/cuda\_scheduling\_examiner\_mirror](https://github.com/yalue/cuda_scheduling_examiner_mirror).
It should work on any CUDA-capable GPU with the Kepler/Maxwell/Pascal/Volta
architecture, such as an NVIDIA GeForce GTX 860M, 1050, 1070, which we ran our
experiments on.  You will also need CUDA 8.0 or 9.0, and
[VisionWorks](https://developer.nvidia.com/embedded/visionworks).

In this [paper (pdf link)](https://cs.unc.edu/~anderson/papers/ecrts18c.pdf),
we have two sets of experiments:

1. Testing of the blocking resources
2. VisionWorks case studies under different scenarios

The following document will walk you through how to install and run the experiments.

Install
========

After downloading the repository from
[GitHub:yalue/cuda\_scheduling\_examiner\_mirror](https://github.com/yalue/cuda_scheduling_examiner_mirror),
follow the [``Compilation``
section](https://github.com/yalue/cuda_scheduling_examiner_mirror#compilation),
or simply ``cd`` into the repository and run ``make``.  Please make sure
``nvcc`` command is available on your ``PATH``.

To compile programs of VisionWorks, run ``make visionworks``.  Please make sure
you have installed VisionWorks.

Evaluate
========

Before running any experiments, be sure to turn off the display service, e.g.,
by running ``sudo service lightdm stop``.  Using remote connection and
disconnecting the display does not ensure the display service to be off.

## Experiment 1: Testing Blocking Resources

We conducted this experiment on Jetson TX2 to generate the plots in section 3
and Table 3 in the paper.  Due to different number of SMs, experiments may
produce different plots on other GPUs.  But the observations and claims stay
consistent.

### Run

We have one script packaging all the cases in Table 3 in the paper.  To get the
data results, run ``python scripts/test_blocking_sources.py``.  To view the
scheduling timeline plots, run ``python scripts/view_blocksbysm.py``.  Note
that this ``view_blocksbysm.py`` cannot process results from experiment 2
below.  It is suggested to have the results from this experiment (1)
self-contained in a separate directory, and pass the director parameter to the
script by ``python scripts/view_blocksbysm.py ./path/to/results/``.

### Compare

To understand the plots and compare it with the results in the paper, we
explain the invocation order of the GPU tasks first.

The launch calls of Task 1, 2, and Task 4 are invocated at time 0, 0.2, and 0.8
seconds, respectively.  Potential synchronization sources under test are
inserted in stream 3, i.e., the stream of Task 3,  at time 0.4 seconds. Task 3
is launched at the time 0.2 seconds after the completion of the inserted
synchronization source.

The plot indicates that the source **does not cause blocking** if all kernels
start right after their launch, same as the ``No blocking`` case where no
synchronization source is inserted.  In this category, we have
``cudaMemcpyAsync (D-D, D-H, or H-D)`` and ``cudaMemsetAsync``.

The plot indicates that the source **blocks other CPU tasks** if the launch of
Task 4 is postponed, demonstrated with the long green arrow.  In this category,
we have ``cudaFree`` and ``cudaFreeHost``.

The plot indicates that the source **causes implicit synchronization** if Task
3 and 4 is postponed.  In this category, we have ``cudaFree``,
``cudaFreeHost``, and ``cudaMemset (sync.)``.

The plot indicates that the source **makes the caller wait for GPU** if Task 3
starts 0.2 seconds after the completion of Task 2.  Because the source starts
and finishes quickly after Task 2 finishes.  And there's 0.2 seconds delay
between the completion of the source and the launch of Task 3.  In this
category, we have ``cudaDeviceSynchronize``, ``cudaFree``, ``cudaFreeHost``,
and ``cudaMemset (sync.)``.  ``cudaStreamSynchronize`` also makes the caller
wait for GPU but it's not demonstrated in the plot.

## Experiment 2: VisionWorks Case Study

In this case study, we run multiple VisionWorks sample programs under different
scenarios (section 4).  We conducted our experiments on GTX 860M, 1050, and
1070.

### Run and Compare

To get the result data, run ``./scripts/visionworks_simplified.sh``.  To
generate plots, run ``python scripts/view_times_pdf.py`` and ``python
scripts/view_times_cdf.py``.  For example,

```bash
./scripts/visionworks_simplified.sh
python scripts/view_times_cdf.py -k "execute_times" -r "hough*x4_0.json 	# Figure 5 in the paper
python scripts/view_times_pdf.py -k "execute_times" -r "hough*x4_0.json 	# Figure 6
python scripts/view_times_cdf.py -k "execute_times" -r "feature*x4_0.json 	# Figure 7
python scripts/view_times_pdf.py -k "execute_times" -r "feature*x4_0.json 	# Figure 8
```
