#pragma once

#ifndef CUDA_RT_H
#define CUDA_RT_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <algorithm>

using std::shared_ptr;
using std::make_shared;
using std::sqrt;

// macro for getting random uniform
#define RND_UNF (curand_uniform(&threadRandState))

// CUDA function wrapper
#define CUDA_CALL(val) checkCuda( (val), #val, __FILE__, __LINE__)

void checkCuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
    if (result) {
        std::cerr << "[CUDA error]: " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(69);
    }
    return;
}

// constants and utilities
const double c_infinity = std::numeric_limits<double>::infinity();
const double c_pi = 3.1415926535897932385;
__device__ constexpr double C_PI() { return 3.1415926535897932385; }


inline double degreesToRads(double degrees) {
    return degrees * c_pi / 180;
}



// common includes
#include "Ray.h"
#include "VEC3.h"



#define RANDVEC3 VEC3(curand_uniform(threadRandState),curand_uniform(threadRandState),curand_uniform(threadRandState))

__device__ VEC3 randVecSphere(curandState* threadRandState) {
    VEC3 p;
    do {
        p = 2.0f * RANDVEC3 - VEC3(1, 1, 1);
    } while (p.squaredNorm() >= 1.0f);
    return p;
}

__device__ VEC3 randVecDisk(curandState* threadRandState) {
    VEC3 p;
    do {
        p = 2.0f * VEC3(curand_uniform(threadRandState), curand_uniform(threadRandState), 0) - VEC3(1, 1, 0);
    } while (dot(p, p) >= 1.0f);
    return p;
}

#endif