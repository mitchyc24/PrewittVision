#include "kernels.h"
#include <cuda_runtime.h>
#include <stdio.h>



__global__ void convert_to_grayscale(unsigned char* input_image, unsigned char* grayscale_image, unsigned int width, unsigned int height) {
    //TODO: implement grayscale conversion
}


extern "C" void convertToGrayscale(unsigned char* host_input_image, unsigned char* host_grayscale_image, unsigned int width, unsigned int height) {
    const int imageSize = width * height * sizeof(unsigned char);
    const int blockSize = 16;
    const int gridSize = (imageSize + blockSize - 1) / blockSize;

    unsigned char* device_input_image;
    unsigned char* device_grayscale_image;

    // CUDA memory operations
    cudaError_t err;
    err = cudaMalloc((void**)&device_input_image, imageSize * 4); // For RGBA
    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating device_input_image: %s\n", cudaGetErrorString(err));
    }

    err = cudaMalloc((void**)&device_grayscale_image, imageSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating device_grayscale_image: %s\n", cudaGetErrorString(err));
    }

    err = cudaMemcpy(device_input_image, host_input_image, imageSize * 4, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying to device_input_image: %s\n", cudaGetErrorString(err));
    }

    convert_to_grayscale<<<gridSize, blockSize>>>(device_input_image, device_grayscale_image, width, height); // Assuming kernel takes width and height
    cudaDeviceSynchronize();

    err = cudaMemcpy(host_grayscale_image, device_grayscale_image, imageSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying to host_grayscale_image: %s\n", cudaGetErrorString(err));
    }

    cudaFree(device_input_image);
    cudaFree(device_grayscale_image);
}
