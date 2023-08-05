#include "kernels.h"
#include <cuda_runtime.h>
#include <stdio.h>



__global__ void apply_prewitt(unsigned char* grayscale_image, unsigned char* output_image, unsigned int width, unsigned int height) {
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




extern "C" void applyPrewitt(unsigned char* host_grayscale_image, unsigned char* host_output_image, unsigned int width, unsigned int height) {
    const int imageSize = width * height * sizeof(unsigned char);
    const dim3 blockSize(16, 16); // Change this as per your needs
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

    apply_prewitt<<<gridSize, blockSize>>>(device_grayscale_image, device_output_image, width, height);
    cudaDeviceSynchronize();

    err = cudaMemcpy(host_output_image, device_output_image, imageSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying to host_output_image: %s\n", cudaGetErrorString(err));
    }

    cudaFree(device_grayscale_image);
    cudaFree(device_output_image);
}


