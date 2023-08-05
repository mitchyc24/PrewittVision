# PrewittVision
PrewittVision is an efficient image processing application that leverages parallel programming techniques using CUDA to perform accelerated edge detection on .png images. By implementing the Prewitt Operator on blocks of pixels in parallel, PrewittVision delivers high-performance gradient image generation, highlighting detected edges.


Initialize PrewittVision project

This commit introduces the initial structure of the PrewittVision project. The application is a C/CUDA program designed to perform edge detection on .png images using the Prewitt operator.

The project contains the following key files:

- main.c: Will contain the primary program execution flow, reading an input image, and writing the processed output image.
- grayscale.cu: Contains the CUDA kernel and associated host code for converting the input image to grayscale.
- prewitt.cu: Contains the CUDA kernel and associated host code for applying the Prewitt operator for edge detection.
- kernels.h: Contains declarations for CUDA functions.
- lodepng.c and lodepng.h: Libraries for handling PNG image input and output.
- Makefile: Used to compile and link the C and CUDA source files into the final executable.

In future commits, the input images will be assumed to be in .png format and will be processed one at a time. The application will use command-line arguments for input and output file paths, and it will perform basic error handling for invalid input.
