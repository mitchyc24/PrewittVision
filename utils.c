#include <string.h>
#include <stdio.h>
#include "utils.h"
#include "lodepng.h"

void encode_and_save(const char* operation, const char* base_name, unsigned char* image, int width, int height) {
    char filename[256];
    construct_filename(base_name, filename, sizeof(filename), operation);

    if (lodepng_encode_file(filename, image, width, height, LCT_GREY, 8)) {
        fprintf(stderr, "Error writing %s image\n", operation);
    } else {
        printf("Created %s\n", filename);
    }
}

const char* extract_base_name(const char* filename, char* base_name_copy, size_t size) {
    const char* base_name = strrchr(filename, '/');
    if (base_name == NULL) {
        base_name = filename;
    } else {
        base_name++;
    }
    strncpy(base_name_copy, base_name, size - 1);
    char* dot = strrchr(base_name_copy, '.');
    if (dot != NULL) {
        *dot = '\0';
    }
    return base_name_copy;
}

void construct_filename(const char* base_name, char* output_filename, size_t size, const char* operation) {
    snprintf(output_filename, size, "output/%s_%s.png", base_name, operation);
}
