#ifdef __cplusplus
extern "C" {
#endif

float convertToGrayscale(unsigned char* host_input_image, unsigned char* host_grayscale_image, unsigned int width, unsigned int height, unsigned int bSize);
float applyPrewitt(unsigned char* host_grayscale_image, unsigned char* host_output_image, unsigned int width, unsigned int height, unsigned int bSize);

#ifdef __cplusplus
}
#endif
