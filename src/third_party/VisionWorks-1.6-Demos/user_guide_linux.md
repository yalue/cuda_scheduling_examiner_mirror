Copyright (c) 2014-2016, NVIDIA CORPORATION.  All rights reserved.

@cond !NVX_DOCS_EXPERIMENTAL
Linux: Building and Running Samples and Demos
@brief Shows how to build samples and demos using native or cross compilation on Linux.

## Native Compilation of Sample Applications and Demos ##

The only method you can use to build the sample applications and demos is a
native build on the target system.

Sources for all samples and demos are provided in the `libvisionworks-samples` package.
After the package installation, source code and make files are located in the
`/usr/share/visionworks/sources` directory. The directory is protected from changes,
so you need to copy its content to any directory with write access.
Here we use your home folder as an example.
All samples use `make` as a build tool.

    $ /usr/share/visionworks/sources/install-samples.sh ~/
    $ cd ~/VisionWorks-<ver>-Samples/
    $ make -j4 # add dbg=1 to make debug build

You can build an individual sample from its directory but the executable will
not be created there nor in a sub-directory. The executable is created in the same directory
as when all samples are built from the top-level directory.

## Running Samples and Demos ##

**Applies to:** ARM devices only. Start the X window manager:

    $ export DISPLAY=:0
    $ X -ac &
    $ blackbox $

Go to the samples directory:

    $ cd ~/VisionWorks-<ver>-Samples/sources/bin/[arch]/linux/release

Run each sample of interest by using the name of the sample. For example, to run `nvx_demo_feature_tracker`, execute:

    $ ./nvx_demo_feature_tracker

@endcond