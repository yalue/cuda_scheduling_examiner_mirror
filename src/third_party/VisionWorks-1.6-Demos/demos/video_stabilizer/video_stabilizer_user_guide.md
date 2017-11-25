Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

Video Stabilizer Demo App
@brief Video Stabilizer Demo user guide.

## Introduction ##

`nvx_demo_video_stabilizer` is a demo that demonstrates the image based video stabilization algorithm.
It uses the Harris feature detector and sparse pyramidal optical flow method (Lucas-Kanade) to estimate a frame's motion.

The demo uses the following pipeline:

                                    (next frame)
                                          |
    +-------------------------------------+-------------------------------+-------+
    |                                     |                               |       |
    |                              [ColorConvert]                         |       |
    |                                     |                               |       |
    |    +--------------------------------+                               |       |
    |    |                                |                               |       |
    |    |                        [GaussianPyramid]                       |       |
    |    |                                |                               |       |
    |    |         (pyr delay -1)   (pyr delay 0)      (pts delay -1)     |       |
    |    |               |                |                  |            |       |
    |    |               |                |                  +----+       |       |
    |    |               |                |                  |    |       |       |
    |    |               +----------------+------------------+    |       |       |
    |    |                                |                       |       |       |
    |    |                         [OpticalFlowPyrLK]             |       |       |
    |    |                                |                       |       |       |
    |    +-------+------------------------+                       |       |       |
    |            |                        |                       |       |       |
    |      [HarrisTrack]                  +-------+---------------+       |       |
    |            |                                |                       |       |
    |      (pts delay 0)                  [FindHomography]                |       |
    |                                             |                       |       |
    |                                     +-------+-----------------------+       |
    |                                     |                                       |
    |                            [HomographyFilter]                               |
    |                                     |                                       |
    |   (...)   (matrix delay -1)   (matrix delay 0)                              |
    |    |             |                  |                                       |
    |    +-------------+------------------+                                  [ImageCopy]
    |                  |                                                          |
    |          [MatrixSmoother]                  (RGBX delay -n)   (...)   (RGBX delay 0)
    |                  |                               |
    +------------------+                               |
                       |                               |
            [TruncateStabTransform]                    |
                       |                               |
                       +---------------+---------------+
                                       |
                               [WarpPerspective]
                                       |
                                 (stabilized)

`nvx_demo_video_stabilizer` is installed in the following directory:

    /usr/share/visionworks/sources/demos/video_stabilizer

For the steps to build sample applications, see the see: nvx_samples_and_demos section for your OS.

## Executing the Video Stabilizer Demo ##

    ./nvx_demo_video_stabilizer [options]

### Command Line Options ###

#### \-s, \--source ####
- Parameter: [Input URI]
- Description: Specifies the input URI. Accepted parameters include a video (.avi), an image sequence (.png, .jpg, .jpeg, .bmp, .tiff) or camera to grab frames.
- Usage:
  - `--source=/path/to/video.avi` for video
  - `--source=/path/to/image_%04d_sequence.png` for image sequence
  - `--source="device:///v4l2?index=0"` for the first V4L2 camera
  - `--source="device:///v4l2?index=1"` for the second V4L2 camera.
  - `--source="device:///nvcamera?index=0"` for the GStreamer NVIDIA camera (Jetson TX1 only).
  - `--source="device:///nvmedia?config=dvp-ov10635-yuv422-ab-e2379&number=4"` for the GStreamer NVIDIA camera (Vibrante for Linux only).

#### \-n ####
- Parameter: [Number of smoothing frames]
- Description: Specifies the number of smoothing frames, should be in the range [1,6] (5 by default). Frames for smoothing are taken from the interval [-numOfSmoothingFrames; numOfSmoothingFrames] in the current frame's vicinity.
- Usage: \n
  `./nvx_demo_video_stabilizer --source=video.avi -n6`

#### \--crop ####
- Parameter: [Crop margin for stabilized frames]
- Description: Specifies a proportion of the width (height) of the frame that is allowed to be cropped for stabilization of the frames. The value should be less than 0.5. If it is negative then the cropping procedure is turned off.
- Usage: \n
  `./nvx_demo_video_stabilizer --source=video.avi --crop=0.1`

#### \-h, \--help ####
- Description: Prints the help message.

### Operational Key ###
- Use `ESC` to close the demo.
- Use `Space` to pause/resume the demo.

