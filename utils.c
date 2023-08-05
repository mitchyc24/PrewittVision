#include <string.h>
#include <stdio.h>
#include "utils.h"


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

void construct_grayscale_filename(const char* base_name, char* output_filename, size_t size) {
    snprintf(output_filename, size, "output/%s_grayscale.png", base_name);
}

void construct_prewitt_filename(const char* base_name, char* output_filename, size_t size) {
    snprintf(output_filename, size, "output/%s_prewitt.png", base_name);
}
