#include "kernels.h"
#include "lodepng.h"
#include "utils.h" 
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Correct usage: %s <input.png>\n", argv[0]);
        return 1;
    }

    const char* filename = argv[1];
    unsigned char* host_input_image;
    unsigned int width, height;

    if (lodepng_decode32_file(&host_input_image, &width, &height, filename)) {
        fprintf(stderr, "Error reading image file %s\n", filename);
        free(host_input_image);
        return 1;
    }

    unsigned char* host_output_image = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    unsigned char* host_grayscale_image = (unsigned char*)malloc(width * height * sizeof(unsigned char));

    convertToGrayscale(host_input_image, host_grayscale_image, width, height);

    char base_name_copy[256];
    const char* base_name = extract_base_name(filename, base_name_copy, sizeof(base_name_copy));

    char grayscale_filename[256];
    construct_grayscale_filename(base_name, grayscale_filename, sizeof(grayscale_filename));

    if (lodepng_encode_file(grayscale_filename, host_grayscale_image, width, height, LCT_GREY, 8)) {
        fprintf(stderr, "Error writing grayscale image\n");
    } else {
        printf("Created %s\n", grayscale_filename);
    }

    applyPrewitt(host_grayscale_image, host_output_image, width, height);

    char prewitt_filename[256];
    construct_prewitt_filename(base_name, prewitt_filename, sizeof(prewitt_filename));

    if (lodepng_encode_file(prewitt_filename, host_output_image, width, height, LCT_GREY, 8)) {
        fprintf(stderr, "Error writing output image\n");
    } else {
        printf("Created %s\n", prewitt_filename);
    }

    free(host_input_image);
    free(host_output_image);
    free(host_grayscale_image);

    return 0;
}
