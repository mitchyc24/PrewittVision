#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>

const char* extract_base_name(const char* filename, char* base_name_copy, size_t size);
void construct_grayscale_filename(const char* base_name, char* output_filename, size_t size);
void construct_prewitt_filename(const char* base_name, char* output_filename, size_t size);

#endif // UTILS_H
