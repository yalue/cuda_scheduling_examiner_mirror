Copyright (c) 2014-2016, NVIDIA CORPORATION.  All rights reserved.

Motion Estimation Demo App
@brief Motion Estimation Demo user guide.

## Introduction ##

`nvx_demo_motion_estimation` is a code sample that implements the NVIDIA Iterative Motion Estimation (IME) algorithm.
IME is a block based motion estimation algorithm which incorporates iterative refinement steps to improve output motion field.

`nvx_demo_motion_estimation` sample pipeline illustrates one-directional motion estimation
computing backward motion vectors from current to previous frame.
The sample pipeline generates a motion vector per every 2x2 block stored in Q14.2 format.

@note Developers may choose to extend the pipeline to implement bidirectional motion estimation.

The following block diagram illustrates steps of the pipeline:

                                     (next frame)
                                           |
                                   [ColorConvertNode]
                                           |
                                 [GaussianPyramidNode]
                                           |
    (current pyramid)               (next pyramid)
             |                             |
             +--------------+--------------+
                            |
                          [IME]
                            |
                     (motion field)

The sample uses vx_delay object to keep these frames from input video.
After graph processing, the delay is aged, the next frame becomes current.

The IME algorithm applies the following pipeline for each pyramid level, starting from the smallest one:

    (motion field from previous level)        (level from current pyramid)            (level from next pyramid)
                   |                                       |                                       |
                   +---------------------------------------+---------------------------------------+
                                                           |
                                               [CreateMotionFieldNode]
                                                           |
                                            (motion field for 8x8 blocks)
                                                           |
                                               [RefineMotionFieldNode]
                                                           |
                                            (motion field for 8x8 blocks)
                                                           |
                                             [PartitionMotionFieldNode]
                                                           |
                                            (motion field for 4x4 blocks)
                                                           |
                                                  [MultiplyByScalar]
                                                           |
                                           (motion field for the next level)

At the end the following pipeline is used:

    (motion field 4x4 for the level 0)
                    |
       [PartitionMotionFieldNode]
                    |
     (motion field for 2x2 blocks)
                    |
           [MultiplyByScalar]
                    |
          (final motion field)

`nvx_demo_motion_estimation` is installed in the following directory:

    usr/share/visionworks/sources/demos/motion_estimation

For the steps to build sample applications, see the see: nvx_samples_and_demos section for your OS.

## Executing Motion Estimation Sample ##

    ./nvx_demo_motion_estimation [options]

### Command Line Options ###

This topic provides a list of supported options and the values they consume.

#### \-s, \--source ####
- Parameter: [inputUri]
- Description: Specifies the input URI. Accepted parameters include a video (in .avi format), an image or an image sequence (in .png, .jpg, .jpeg, .bmp, or .tiff format), or camera.
- Usage:
  - `--source=/path/to/video.avi` for video
  - `--source=/path/to/image` for image
  - `--source=/path/to/image_%04d_sequence.png` for image sequence
  - `--source="device:///v4l2?index=0"` for the first V4L2 camera
  - `--source="device:///v4l2?index=1"` for the second V4L2 camera.
  - `--source="device:///nvcamera?index=0"` for the GStreamer NVIDIA camera (Jetson TX1 only).

#### \-c, \--config ####
- Parameter: [Config file path]
- Description: Specifies the path to the configuration file. The file contains the parameters
  of the algorithm stored in key=value format. Note that the config file contains information
  on the intrinsic parameters of the camera, so using the default config file for different
  videos may sometimes give a result with insufficient quality.

    This file contains the following parameters:

    - **biasWeight**
        - Parameter: [floating point value greater than or equal to 0]
        - Description: The weight of bias distance in Create Motion Field primitive. Default is 1.0.

    - **mvDivFactor**
        - Parameter: [integer value greater than or equal to 0 and less than or equal to 16]
        - Description: mvDivFactor specifies the minimum Manhattan distance of the second best motion vector
                       from the best motion vector selected for a block.
                       Having mvDivFactor imposes selection of a second diverse motion vector
                       to represent the blocks located near the object boundaries.
                       Default is 4.

    - **smoothnessFactor**
        - Parameter: [floating point value greater than or equal to 0]
        - Description: The smoothness factor for motion field. Default is 1.0.

#### -h, \--help ####
- Parameter: true
- Description: Prints the help message.

### Operational Key ###
- Use `Space` to pause/resume the demo.
- Use `ESC` to close the demo.

