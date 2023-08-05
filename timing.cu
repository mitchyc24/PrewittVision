#include "timing.h"
#include <stdio.h>

void timeKernelExecution(const std::function<void()>& kernelFunction) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    kernelFunction(); // Execute the kernel function
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedMilliseconds = 0;
    cudaEventElapsedTime(&elapsedMilliseconds, start, stop);

    printf("Kernel execution time: %.3f ms\n", elapsedMilliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
