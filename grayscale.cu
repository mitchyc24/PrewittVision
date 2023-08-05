#include "kernels.h"
#include <cuda_runtime.h>


__global__ void convert_to_grayscale(unsigned char* input_image, unsigned char* grayscale_image) {
    //TODO: implement grayscale conversion
}


void convertToGrayscale(unsigned char* host_input_image, unsigned char* host_grayscale_image) {
    const int imageSize = 1024 * 1024;
    const int blockSize = 16;
    const int gridSize = imageSize / blockSize;

    unsigned char* device_input_image;
    unsigned char* device_grayscale_image;

    // CUDA memory operations
    cudaMalloc((void**)&device_input_image, imageSize * 4); // For RGBA
    cudaMalloc((void**)&device_grayscale_image, imageSize);

    cudaMemcpy(device_input_image, host_input_image, imageSize * 4, cudaMemcpyHostToDevice);

    convert_to_grayscale<<<gridSize, blockSize>>>(device_input_image, device_grayscale_image);
    cudaDeviceSynchronize();

    cudaMemcpy(host_grayscale_image, device_grayscale_image, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(device_input_image);
    cudaFree(device_grayscale_image);
}
