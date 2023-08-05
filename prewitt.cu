#include "kernels.h"
#include <cuda_runtime.h>
#include <stdio.h>



__global__ void apply_prewitt(unsigned char* grayscale_image, unsigned char* output_image, unsigned int width, unsigned int height) {
    //TODO: implement apply_prewitt 
}



extern "C" void applyPrewitt(unsigned char* host_grayscale_image, unsigned char* host_output_image, unsigned int width, unsigned int height) {
    const int imageSize = width * height * sizeof(unsigned char);
    const int blockSize = 16;
    const int gridSize = (imageSize + blockSize - 1) / blockSize;

    unsigned char* device_grayscale_image;
    unsigned char* device_output_image;

    // CUDA memory operations
    cudaError_t err;
    err = cudaMalloc((void**)&device_grayscale_image, imageSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating device_grayscale_image: %s\n", cudaGetErrorString(err));
    }

    err = cudaMalloc((void**)&device_output_image, imageSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating device_output_image: %s\n", cudaGetErrorString(err));
    }

    err = cudaMemcpy(device_grayscale_image, host_grayscale_image, imageSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying to device_grayscale_image: %s\n", cudaGetErrorString(err));
    }

    apply_prewitt<<<gridSize, blockSize>>>(device_grayscale_image, device_output_image, width, height);
    cudaDeviceSynchronize();

    err = cudaMemcpy(host_output_image, device_output_image, imageSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying to host_output_image: %s\n", cudaGetErrorString(err));
    }

    cudaFree(device_grayscale_image);
    cudaFree(device_output_image);
}


