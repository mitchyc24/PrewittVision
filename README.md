# PrewittVision

PrewittVision is an efficient image processing application that leverages parallel programming techniques using CUDA to perform accelerated edge detection on .png images. By implementing the Prewitt Operator on blocks of pixels in parallel, PrewittVision delivers high-performance gradient image generation, highlighting detected edges.

## Components

- `main.c`: Contains the primary program execution flow, reading an input image, and writing the processed output image. It also includes parallel processing of images using OpenMP.
- `grayscale.cu`: Contains the CUDA kernel and associated host code for converting the input image to grayscale.
- `prewitt.cu`: Contains the CUDA kernel and associated host code for applying the Prewitt operator for edge detection.
- `kernels.h`: Contains declarations for CUDA functions.
- `lodepng.c` and `lodepng.h`: Libraries for handling PNG image input and output.
- `Makefile`: Used to compile and link the C and CUDA source files into the final executable.
- `timing.cu` and `timing.h`: Contains functions for timing the kernel execution.
- `utils.c` and `utils.h`: Contains utility functions for encoding, saving, and constructing filenames.
- `log_manager.c` and `log_manager.h`: Contains functions for managing log files, including writing and freeing logs.

## How to Use

### Compilation
Use the provided Makefile to compile the source code. Run the make command in the terminal to build the executable.

### Execution
Use the `make run` command to compile (if necessary) and run the executable:

```bash
make run
```
The program will automatically process all the images in the `data` directory. Alternatively, you can run the compiled executable directly:
```bash
./PrewittVision
```
When run directly, it processes images from the `imgs` directory (Note: the standard input directory used by `make run` is `data`).

### Output
The program will process the input images, apply the grayscale conversion, and then apply the Prewitt operator for edge detection. The processed images will be saved in the output directory with appropriate filenames.

### Logging
Timing information for the Grayscale and Prewitt kernels will be logged in the `logs` directory in two files:
- `timing_log.txt`: A human-readable log including version and kernel execution times for different block sizes.
- `timing_log.csv`: A comma-separated values file suitable for data analysis, containing image names, kernel times, and block sizes.

### Code Structure
Grayscale Conversion: The convert_to_grayscale kernel in grayscale.cu is responsible for converting the input image to grayscale using standard weightings for the RGB channels.
Prewitt Operator: The apply_prewitt kernel in prewitt.cu applies the Prewitt operator to detect edges in the grayscale image.
Timing: The timeKernelExecution function in timing.cu is used to measure the execution time of the kernels.
Utilities: Functions in utils.c are used for encoding and saving images, extracting base filenames, and constructing output filenames.
Logging: Functions in log_manager.c are used to manage log files, including writing and freeing logs.

### Dependencies
CUDA
OpenMP
lodepng