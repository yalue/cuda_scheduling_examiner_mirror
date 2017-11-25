Copyright (c) 2015-2016, NVIDIA CORPORATION.  All rights reserved.

Stereo Matching Demo App
@brief Stereo Matching Demo user guide.

## Introduction ##

`nvx_demo_stereo_matching` is a simple stereo matching demo that uses
Semi-Global Matching algorithm to evaluate disparity. It performs color
conversion and downscaling prior to evaluating stereo for better quality and
performance. The input images are expected to be undistorted and rectified.
For more information on how to rectify the stereo pair, see [OpenCV's Stereo Rectifation tutorial](http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#stereorectify).

## Pipeline Details ##

The pipeline can be illustrated by the following diagram:

          (left frame)              (right frame)
               |                          |
          [ScaleImage] (down)        [ScaleImage] (down)
               |                          |
         [ColorConvert] (to gray)   [ColorConvert] (to gray)
               |                          |
               +---------+     +----------+
                         |     |
                   [SemiGlobalMatching]
                            |
                      [ConvertDepth] (to 8-bit)
                            |
                       [ScaleImage] (up)
                            |
                        [Multiply]
                            |
                     (disparity image)

In the second step, the demo converts the disparity image into color output using
the following pipeline:

                     (disparity image)
                             |
          +------------------+------------------+
          |                  |                  |
    [TableLookup]      [TableLookup]      [TableLookup]
          |                  |                  |
    (Red Channel)      (Blue Channel)    (Green Channel)
          |                  |                  |
          +------------------+------------------+
                             |
                     [ChannelCombine]
                             |
                      (output image)

Color output is created using linear conversion of disparity values from [0..ndisp)
interval into the HSV color space, where the smallest disparity (far objects)
corresponds to [H=240, S=1, V=1] (blue color) and the largest disparity (near
objects) corresponds to [H=0, S=1, V=1] (red color). The resulting HSV value
is then converted to RGB color space for visualization.

## Installation and Usage ##

`nvx_demo_stereo_matching` is installed in the following directory:

    /usr/share/visionworks/sources/demos/stereo_matching

For the steps to build sample applications, see the see: nvx_samples_and_demos
section for your OS.

## Executing the Stereo Matching Demo ##

    ./nvx_demo_stereo_matching [options]

### Command Line Options ###

This topic provides a list of supported options and the values they consume.

#### \-s, \--source ####
- Parameter: [input URI]
- Description: Specifies the input URI. Video, image, or image sequence must
  contain both channels in top-bottom layout.
- Usage:

    - `--source=/path/to/image` for image
    - `--source=/path/to/video.avi` for video
    - `--source=/path/to/image_%04d_sequence.png` for image sequence

#### \-c, \--config ####
- Parameter: [config file path]
- Description: Specifies the path to the configuration file.

    The file contains the parameters of the stereo matching algorithm.
      - **min_disparity**
      - **max_disparity**
          - Parameter: [integer value greater or equal to zero and less or equal
            to 256]
          - Description: Minimum and maximum disparity values. Defaults are 0 and
            64. Maximum disparity must be divisible by 4.

      - **P1**
      - **P2**
          - Parameter: [integer value greater than or equal to zero and less than
            or equal to 256]
          - Description: Penalty parameters for SGBM algorithm. The larger the values,
            the smoother the disparity. P1 is the penalty on the disparity change
            by plus or minus 1 between neighbor pixels; P2 - by more than 1. The
            algorithm requires P2 > P1. Defaults are 8 and 109.

      - **sad**
          - Parameter: [odd integer greater than or equal to zero and less than or equal to 31]
          - Description: The size of the SAD window. Default is 5.

      - **bt_clip_value**
          - Parameter: [odd integer in range 15 to 95]
          - Description: Truncation value for pre-filtering algorithm. It first computes
            x-derivative at each pixel and clips its value to [-bt_clip_value, bt_clip_value]
            interval. Default is 31.

      - **max_diff**
          - Parameter: [integer value greater than or equal to zero]
          - Description: Maximum allowed difference (in integer pixel units) in the
            left-right disparity check. Default is 32000.

      - **uniqueness_ratio**
          - Parameter: [integer value greater than or equal to zero and less than
            or equal to 100]
          - Description: Margin in percents by which the best (minimum) computed
            cost function value must beat the second best value to consider the
            found match correct.

      - **scanlines_mask**
          - Parameter: [integer value in range 0 to 255]
          - Description: Bit-mask for enabling any combination of 8 possible directions.
            The lowest bit corresponds to "from-left-to-right" direction
            (NVX_SCANLINE_LEFT_RIGHT enumeration value). The second lowest bit corresponds
            to "from-top-left-to-bottom-right" direction (NVX_SCANLINE_TOP_LEFT_BOTTOM_RIGHT
            enumeration value), and so on. Default is 255.

      - **flags**
          - Parameter: [integer value in range 0 to 3]
          - Description: Extra flags for SGBM algorithm. Default is 0 which implies no cost filtration.

      - **ct_win_size**
          - Parameter: [odd integer greater than or equal to zero and less than or equal to 31]
          - Description: The size of the Census Transform window. Default is 0.

      - **hc_win_size**
          - Parameter: [odd integer greater than or equal to zero and less than or equal to 31]
          - Description: The size of the Hamming Cost window. Default is 0.

- Usage:

  `./nvx_demo_stereo_matching --config=/path/to/config_file.ini`

- If the argument is omitted, the default config file will be used.

#### \-t, \--type ####
- Parameter: hl, ll, pyr
- Description: Specifies the stereo pipeline implementation type.
- Usage:
    - `--type=hl` chooses the implementation via high-level API
    - `--type=ll` chooses the implementation via low-level API
    - `--type=pyr` chooses the implementation via low-level API organized in a
      pyramidal scheme

#### -h, \--help ####
- Description: Prints the help message.

### Operational Keys ###
- Use `S` to switch between displaying the original frame, disparity image, and color output.
- Use `Space` to pause/resume the demo.
- Use `ESC` to close the demo.

