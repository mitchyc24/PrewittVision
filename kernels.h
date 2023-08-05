#ifdef __cplusplus
extern "C" {
#endif

void convertToGrayscale(unsigned char* host_input_image, unsigned char* host_grayscale_image);
void applyPrewitt(unsigned char* host_grayscale_image, unsigned char* host_output_image);

#ifdef __cplusplus
}
#endif
