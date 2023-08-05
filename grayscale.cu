#include "kernels.h"
#include "timing.h"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void convert_to_grayscale(unsigned char* input_image, unsigned char* grayscale_image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < width && y < height) {
        int pixelIndex = y * width + x;
        unsigned char r = input_image[4 * pixelIndex];
        unsigned char g = input_image[4 * pixelIndex + 1];
        unsigned char b = input_image[4 * pixelIndex + 2];

        // Calculate the grayscale value
        grayscale_image[pixelIndex] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

extern "C" float convertToGrayscale(unsigned char* host_input_image, unsigned char* host_grayscale_image, unsigned int width, unsigned int height) {
    const int imageSize = width * height * sizeof(unsigned char);
    const int blockSize = 16;
    dim3 blockDims(blockSize, blockSize, 1);
    dim3 gridDims((width + blockDims.x - 1) / blockDims.x, (height + blockDims.y - 1) / blockDims.y, 1);

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

    auto kernelFunction = [&]() {
        convert_to_grayscale<<<gridDims, blockDims>>>(device_input_image, device_grayscale_image, width, height);
    };

    // Call and time the kernel execution using the timing function
    printf("Launching Grayscale Kernel\n");
    float elapsedTime = timeKernelExecution(kernelFunction);


    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error launching kernel: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    err = cudaMemcpy(host_grayscale_image, device_grayscale_image, imageSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying to host_grayscale_image: %s\n", cudaGetErrorString(err));
    }

    cudaFree(device_input_image);
    cudaFree(device_grayscale_image);

    return elapsedTime;
}
