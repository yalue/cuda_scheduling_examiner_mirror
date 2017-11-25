Copyright (c) 2014-2016, NVIDIA CORPORATION.  All rights reserved.

Hough Transform Demo App
@brief Hough Transform Demo user guide.

## Introduction ##

`nvx_demo_hough_transform` demonstrates lines and circles detection via Hough Transform.

## Details of the pipeline ##

The demo uses the following pipeline:

                                   (frame)
                                      |
                                [ColorConvert]
                                      |
                               [ChannelExtract]
                                      |
                             [ScaleImage (down)]
                                      |
                                 [Median3x3]
                                      |
                                [EqualizeHist]
                                      |
               +----------------------+----------------------+
               |                                             |
       [CannyEdgeDetector]                                   |
               |                                             |
           +---+---------------------+----+              [Sobel3x3]
           |                         |    |                  |
           |                         |    +------------------+
           |                         |                       |
    [ScaleImage (up)]         [HoughSegments]         [HoughCircles]
           |                         |                       |
        (edges)                 (segments)               (circles)

The input frame is converted to grayscale, downscaled, blurred with Median filter,
and equalized. The equalized frame is then processed by Canny Edge Detector and Sobel operator, and
the resulting edges image and derivatives are passed to the Hough Circle node to
get the final array with detected circles. The edges image is also passed to the
Hough Segments node to get the final array with detected lines.

## Installation and Usage ##

`nvx_demo_hough_transform` is installed in the following directory:

    /usr/share/visionworks/sources/demos/hough_transform

For the steps to build sample applications, see the see: nvx_samples_and_demos section for your OS.

## Executing the Hough Transform Demo ##

    ./nvx_demo_hough_transform [options]

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
- Parameter: [config file path]
- Description: Specifies the path to the configuration file.

    This file contains the following parameters:

    - **switchPeriod**
        - Parameter: [integer value greater than or equal to zero]
        - Description: The period in frames between source/edges auto switch (0 - no auto switch). The default is 400.

    - **scaleFactor**
        - Parameter: [floating point value greater than 0 and less than or equal to 1]
        - Description: [ScaleImage] The scale factor. Default is 0.5.

    - **scaleType**
        - Parameter: [nearest, bilinear, or area]
        - Description: [ScaleImage] The scale interpolation type. Default is bilinear.

    - **CannyLowerThresh**
        - Parameter: [integer value greater than zero]
        - Description: [CannyEdgeDetector] The lower threshold. Default is 230.

    - **CannyUpperThresh**
        - Parameter: [integer value greater than zero]
        - Description: [CannyEdgeDetector] The upper threshold. Default is 250.

    - **dp**
        - Parameter: [floating point value greater than or equal to 1]
        - Description: [HoughCircles] Inverse ratio of the accumulator resolution to the image resolution for the downscaled frame. Default is 2.

    - **minDist**
        - Parameter: [floating point value greater than zero]
        - Description: [HoughCircles] Minimum distance between the centers of the detected circles for the downscaled frame. Default is 10.

    - **minRadius**
        - Parameter: [integer value greater than zero]
        - Description: [HoughCircles] Minimum circle radius for the downscaled frame. Default is 1.

    - **maxRadius**
        - Parameter: [integer value greater than zero]
        - Description: [HoughCircles] Maximum circle radius for the downscaled frame. Default is 25.

    - **accThreshold**
        - Parameter: [integer value greater than zero]
        - Description: [HoughCircles] The accumulator threshold for the circle centers at the detection stage for the downscaled frame. Default is 110.

    - **circlesCapacity**
        - Parameter: [integer value greater than zero and less than or equal to 1000]
        - Description: [HoughCircles] The capacity of output array for detected circles. Default is 300.

    - **rho**
        - Parameter: [float value greater than zero]
        - Description: [HoughSegments] The distance resolution of the accumulator in pixels for the downscaled frame. Default is 1.0.

    - **theta**
        - Parameter: [float value greater than zero and less then 180]
        - Description: [HoughSegments] The angle resolution of the accumulator in degrees for the downscaled frame. Default is 1.0.

    - **votesThreshold**
        - Parameter: [integer value greater than zero]
        - Description: [HoughSegments] The accumulator threshold parameter for the downscaled frame. Default is 100.

    - **minLineLength**
        - Parameter: [integer value greater than zero]
        - Description: [HoughSegments] The minimum line length for the downscaled frame. Default is 25.

    - **maxLineGap**
        - Parameter: [integer value greater than or equal t0 zero]
        - Description: [HoughSegments] The maximum allowed gap between points on the same line for the downscaled frame. Default is 2.

    - **linesCapacity**
        - Parameter: [integer value greater than zero and less than or equal to 1000]
        - Description: [HoughSegments] The capacity of output array for detected lines. Default is 300.

- Usage:

  `./nvx_demo_hough_transform --config=/path/to/config_file.ini`

- If the argument is omitted, the default config file will be used.

#### \-h, \--help ####
- Description: Prints the help message.

### Operational Keys ###
- Use `M` to switch source/edges.
- Use `Space` to pause/resume the demo.
- Use `ESC` to close the demo.

