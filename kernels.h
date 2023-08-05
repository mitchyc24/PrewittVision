#ifdef __cplusplus
extern "C" {
#endif

void convertToGrayscale(unsigned char* host_input_image, unsigned char* host_grayscale_image, unsigned int width, unsigned int height);
void applyPrewitt(unsigned char* host_grayscale_image, unsigned char* host_output_image, unsigned int width, unsigned int height);

#ifdef __cplusplus
}
#endif
