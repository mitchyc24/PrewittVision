#include "kernels.h"
#include <cuda_runtime.h>


__global__ void apply_prewitt(unsigned char* grayscale_image, unsigned char* output_image) {
    //TODO: implement apply_prewitt 
}



void applyPrewitt(unsigned char* host_grayscale_image, unsigned char* host_output_image) {
    const int imageSize = 1024 * 1024;
    const int blockSize = 16;
    const int gridSize = imageSize / blockSize;

    unsigned char* device_grayscale_image;
    unsigned char* device_output_image;

    // CUDA memory operations
    cudaMalloc((void**)&device_grayscale_image, imageSize);
    cudaMalloc((void**)&device_output_image, imageSize);

    cudaMemcpy(device_grayscale_image, host_grayscale_image, imageSize, cudaMemcpyHostToDevice);

    apply_prewitt<<<gridSize, blockSize>>>(device_grayscale_image, device_output_image);
    cudaDeviceSynchronize();

    cudaMemcpy(host_output_image, device_output_image, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(device_grayscale_image);
    cudaFree(device_output_image);
}
