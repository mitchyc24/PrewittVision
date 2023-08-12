# PrewittVision

PrewittVision is an efficient image processing application that leverages parallel programming techniques using CUDA to perform accelerated edge detection on .png images. By implementing the Prewitt Operator on blocks of pixels in parallel, PrewittVision delivers high-performance gradient image generation, highlighting detected edges. 

## Components

- `main.c`: Contains the primary program execution flow, reading an input image, and writing the processed output image. It also includes parallel processing of images using OpenMP.
- `grayscale.cu`: Contains the OpenMP threading, , CUDA kernel and associated host code for converting the input image to grayscale.
- `prewitt.cu`: Contains the OpenMP threading, CUDA kernel and associated host code for applying the Prewitt operator for edge detection.
- `kernels.h`: Contains declarations for CUDA and OpenMP threading functions.
- `lodepng.c` and `lodepng.h`: Libraries for handling PNG image input and output.
- `Makefile`: Used to compile and link the C and CUDA source files into the final executable.
- `timing.cu` and `timing.h`: Contains functions for timing the kernel execution.
- `utils.c` and `utils.h`: Contains utility functions for encoding, saving, and constructing filenames.
- `log_manager.c` and `log_manager.h`: Contains functions for managing log files, including writing and freeing logs.

## How to Use

### Compilation
Use the provided Makefile to compile the source code. Run the make command in the terminal to build the executable.

### Execution
Run the compiled executable with the following command:

```c
./PrewittVision
```
The program will automatically process all the images in the "imgs" directory.

###Output
The program will process the input images, apply the grayscale conversion, and then apply the Prewitt operator for edge detection. The processed images will be saved in the output directory with appropriate filenames.

###Logging
Timing information for the Grayscale and Prewitt kernels will be logged in the timing_log.txt file, including the version and kernel execution time.

###Code Structure
Grayscale Conversion: The convert_to_grayscale kernel in grayscale.cu is responsible for converting the input image to grayscale using standard weightings for the RGB channels.
Prewitt Operator: The apply_prewitt kernel in prewitt.cu applies the Prewitt operator to detect edges in the grayscale image.
Timing: The timeKernelExecution function in timing.cu is used to measure the execution time of the kernels.
Utilities: Functions in utils.c are used for encoding and saving images, extracting base filenames, and constructing output filenames.
Logging: Functions in log_manager.c are used to manage log files, including writing and freeing logs.


### Dependencies
CUDA
OpenMP
lodepng (included in the repository)