#include "kernels.h"
#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Correct usage: %s <input.png>\n", argv[0]);
        return 1;
    }

    const char* filename = argv[1];
    unsigned char* host_input_image;
    unsigned int width, height;

    // Read the input image file using lodepng
    if (lodepng_decode32_file(&host_input_image, &width, &height, filename)) {
        fprintf(stderr, "Error reading image file %s\n", filename);
        return 1;
    }

    unsigned char* host_output_image = (unsigned char*)malloc(width * height);
    unsigned char* host_grayscale_image = (unsigned char*)malloc(width * height);

    convertToGrayscale(host_input_image, host_grayscale_image, width, height);
    applyPrewitt(host_grayscale_image, host_output_image, width, height);

    // TODO: Write host_output_image to a .png file using lodepng

    free(host_input_image);
    free(host_output_image);
    free(host_grayscale_image);

    return 0;
}
