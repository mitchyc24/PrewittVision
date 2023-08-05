#pragma once
#include <functional>
#include <cuda_runtime.h>

void timeKernelExecution(const std::function<void()>& kernelFunction);
