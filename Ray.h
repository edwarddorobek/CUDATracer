
#pragma once

#ifndef RAYH
#define RAYH
#include "CUDA_RT.h"
#include "VEC3.h"

class Ray
{
public:
    __device__ Ray() {}
    __device__ Ray(const VEC3& a, const VEC3& b) : m_origin(a), m_direction(b) {}
    __device__ VEC3 origin() const { return m_origin; }
    __device__ VEC3 direction() const { return m_direction; }
    __device__ VEC3 at(float t) const { return m_origin + t * m_direction; }

    VEC3 m_origin;
    VEC3 m_direction;
};

#endif