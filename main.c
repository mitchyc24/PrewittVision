#include "kernels.h"
#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>

// TODO: Parse command line arguments, read input image file using lodepng, etc.

int main() {

    unsigned char* host_input_image; // TODO: Initialize this with the image data
    unsigned char* host_output_image = (unsigned char*)malloc(1024 * 1024);
    unsigned char* host_grayscale_image = (unsigned char*)malloc(1024 * 1024);

    convertToGrayscale(host_input_image, host_grayscale_image);
    applyPrewitt(host_grayscale_image, host_output_image);

    // TODO: Write host_output_image to a .png file using lodepng

    free(host_input_image);
    free(host_output_image);
    free(host_grayscale_image);

    return 0;
}
