#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>

void encode_and_save(const char* operation, const char* base_name, unsigned char* image, int width, int height);
const char* extract_base_name(const char* filename, char* base_name_copy, size_t size);
void construct_filename(const char* base_name, char* output_filename, size_t size, const char* operation);

#endif // UTILS_H
