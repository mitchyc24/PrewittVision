#include "kernels.h"
#include "timing.h"
#include <sys/time.h>   
#include <cuda_runtime.h>
#include <stdio.h>
#include <omp.h>



__global__ void kernel_prewitt(unsigned char* grayscale_image, unsigned char* output_image, unsigned int width, unsigned int height) {
    // Prewitt operator https://en.wikipedia.org/wiki/Prewitt_operator

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1) return;

    // Prewitt operator kernels
    int Gx[3][3] = { { -1, 0, 1 }, { -1, 0, 1 }, { -1, 0, 1 } };
    int Gy[3][3] = { { -1, -1, -1 }, { 0, 0, 0 }, { 1, 1, 1 } };

    float gradient_x = 0.0f;
    float gradient_y = 0.0f;

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            int pixel = grayscale_image[(y + j) * width + (x + i)];
            gradient_x += pixel * Gx[i + 1][j + 1];
            gradient_y += pixel * Gy[i + 1][j + 1];
        }
    }

    float magnitude = sqrtf(gradient_x * gradient_x + gradient_y * gradient_y);

    // Convert the float to an unsigned char (0 to 255 range)
    unsigned char edge_strength = min(255, (int)magnitude);

    // Set the output pixel value
    output_image[y * width + x] = edge_strength;
}

/* Apply Prewitt operator to grayscale image using threading */
extern "C" float applyPrewittThreading(unsigned char* host_grayscale_image, unsigned char* host_output_image, unsigned int width, unsigned int height, unsigned int threads) {
    printf("Launching Prewitt Operator with Threading\n");
    // Prewitt operator kernels
    int Gx[3][3] = { { -1, 0, 1 }, { -1, 0, 1 }, { -1, 0, 1 } };
    int Gy[3][3] = { { -1, -1, -1 }, { 0, 0, 0 }, { 1, 1, 1 } };

    // Calculate the time it takes to process image with one core
    struct timeval start, stop;
    gettimeofday(&start, NULL);

    #pragma omp parallel for collapse(2) num_threads(threads)
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            float gradient_x = 0.0f;
            float gradient_y = 0.0f;

            // Do not calculate edges
            if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1) {
                host_output_image[y * width + x] = 0;
                continue;
            }
            
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    int pixel = host_grayscale_image[(y + j) * width + (x + i)];
                    gradient_x += pixel * Gx[i + 1][j + 1];
                    gradient_y += pixel * Gy[i + 1][j + 1];
                }
            }

            float magnitude = sqrtf(gradient_x * gradient_x + gradient_y * gradient_y);

            // Convert the float to an unsigned char (0 to 255 range)
            unsigned char edge_strength = min(255, (int)magnitude);

            // Set the output pixel value
            host_output_image[y * width + x] = edge_strength;
        }
    }

    gettimeofday(&stop, NULL);
    printf("Threading execution time: %f ms\n", (float)(stop.tv_usec - start.tv_usec)/1000);

    return (float)(stop.tv_usec - start.tv_usec)/1000;

}

/* Apply Prewitt operator to grayscale image using CUDA */
extern "C" float applyPrewittCuda(unsigned char* host_grayscale_image, unsigned char* host_output_image, unsigned int width, unsigned int height, unsigned int bSize) {
    const int imageSize = width * height * sizeof(unsigned char);
    const dim3 blockSize(bSize, bSize); // Change this as per your needs
    const dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    

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



    auto kernelFunction = [&](){
        kernel_prewitt<<<gridSize, blockSize>>>(device_grayscale_image, device_output_image, width, height);
    };

    // Call and time the kernel execution using the timing function
    printf("Launching Prewitt Operator Kernel\n");
    float elapsedTime = timeKernelExecution(kernelFunction);


    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error launching kernel: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    err = cudaMemcpy(host_output_image, device_output_image, imageSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying to host_output_image: %s\n", cudaGetErrorString(err));
    }

    cudaFree(device_grayscale_image);
    cudaFree(device_output_image);

    return elapsedTime;
}


