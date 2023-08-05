#ifndef KERNELS_H
#define KERNELS_H

// Function declarations
void convertToGrayscale(unsigned char* host_input_image, unsigned char* host_grayscale_image);
void applyPrewitt(unsigned char* host_grayscale_image, unsigned char* host_output_image);

#endif
