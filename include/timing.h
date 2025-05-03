#pragma once
#include <functional>
#include <cuda_runtime.h>

float timeKernelExecution(const std::function<void()>& kernelFunction);
