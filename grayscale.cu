#include "kernels.h"
#include "timing.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>  
#include <omp.h>

__global__ void kernel_grayscale(unsigned char* input_image, unsigned char* grayscale_image, int width, int height) {
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

extern "C" float convertToGrayscaleCuda(unsigned char* host_input_image, unsigned char* host_grayscale_image, unsigned int width, unsigned int height, unsigned int bSize) {
    const int imageSize = width * height * sizeof(unsigned char);
    dim3 blockDims(bSize, bSize, 1);
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
        kernel_grayscale<<<gridDims, blockDims>>>(device_input_image, device_grayscale_image, width, height);
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

extern "C" float convertToGrayscaleThreading(unsigned char* host_input_image, unsigned char* host_grayscale_image, unsigned int width, unsigned int height, unsigned int threads) {
    printf("Launching Grayscale Threading\n");
    // Calculate the time it takes to process image with one core
    struct timeval start, stop;
    gettimeofday(&start, NULL);

    #pragma omp parallel for collapse(2) num_threads(threads)
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            int pixelIndex = y * width + x;
            unsigned char r = host_input_image[4 * pixelIndex];
            unsigned char g = host_input_image[4 * pixelIndex + 1];
            unsigned char b = host_input_image[4 * pixelIndex + 2];

            // Calculate the grayscale value
            host_grayscale_image[pixelIndex] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);

        }
    }

    gettimeofday(&stop, NULL);
    printf("Grayscale execute time with %d thread(s): %f ms\n", threads, (float)(stop.tv_usec - start.tv_usec)/1000);

    return (float)(stop.tv_usec - start.tv_usec)/1000;

}