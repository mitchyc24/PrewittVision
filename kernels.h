#ifdef __cplusplus
extern "C" {
#endif

float convertToGrayscaleCuda(unsigned char* host_input_image, unsigned char* host_grayscale_image, unsigned int width, unsigned int height, unsigned int bSize);
float convertToGrayscaleThreading(unsigned char* host_input_image, unsigned char* host_grayscale_image, unsigned int width, unsigned int height, unsigned int threads);
float applyPrewittCuda(unsigned char* host_grayscale_image, unsigned char* host_output_image, unsigned int width, unsigned int height, unsigned int bSize);
float applyPrewittThreading(unsigned char* host_grayscale_image, unsigned char* host_output_image, unsigned int width, unsigned int height, unsigned int threads);


#ifdef __cplusplus
}
#endif
