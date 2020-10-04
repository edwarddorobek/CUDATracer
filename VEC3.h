#pragma once

#ifndef VEC3_H
#define VEC3_H

#include "CUDA_RT.h"

class VEC3
{
public:
	__host__ __device__ VEC3() {}
    __host__ __device__ VEC3(float e0, float e1, float e2) { e[0] = e0; e[1] = e1; e[2] = e2; }
    __host__ __device__ inline float x() const { return e[0]; }
    __host__ __device__ inline float y() const { return e[1]; }
    __host__ __device__ inline float z() const { return e[2]; }
    __host__ __device__ inline float r() const { return e[0]; }
    __host__ __device__ inline float g() const { return e[1]; }
    __host__ __device__ inline float b() const { return e[2]; }

    __host__ __device__ inline const VEC3& operator+() const { return *this; }
    __host__ __device__ inline VEC3 operator-() const { return VEC3(-e[0], -e[1], -e[2]); }
    __host__ __device__ inline float operator[](int i) const { return e[i]; }
    __host__ __device__ inline float& operator[](int i) { return e[i]; };

    __host__ __device__ inline VEC3& operator+=(const VEC3& v2);
    __host__ __device__ inline VEC3& operator-=(const VEC3& v2);
    __host__ __device__ inline VEC3& operator*=(const VEC3& v2);
    __host__ __device__ inline VEC3& operator/=(const VEC3& v2);
    __host__ __device__ inline VEC3& operator*=(const float t);
    __host__ __device__ inline VEC3& operator/=(const float t);

    __host__ __device__ inline float norm() const { return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]); }
    __host__ __device__ inline float squaredNorm() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }
    __host__ __device__ inline void normalize();

	float e[3];
};



inline std::istream& operator>>(std::istream& is, VEC3& t) {
    is >> t.e[0] >> t.e[1] >> t.e[2];
    return is;
}

inline std::ostream& operator<<(std::ostream& os, const VEC3& t) {
    os << t.e[0] << " " << t.e[1] << " " << t.e[2];
    return os;
}

__host__ __device__ inline void VEC3::normalize() {
    float k = 1.0 / sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
    e[0] *= k; e[1] *= k; e[2] *= k;
}

__host__ __device__ inline VEC3 operator+(const VEC3& v1, const VEC3& v2) {
    return VEC3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__host__ __device__ inline VEC3 operator-(const VEC3& v1, const VEC3& v2) {
    return VEC3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__host__ __device__ inline VEC3 operator*(const VEC3& v1, const VEC3& v2) {
    return VEC3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__host__ __device__ inline VEC3 operator/(const VEC3& v1, const VEC3& v2) {
    return VEC3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__host__ __device__ inline VEC3 operator*(float t, const VEC3& v) {
    return VEC3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline VEC3 operator/(VEC3 v, float t) {
    return VEC3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
}

__host__ __device__ inline VEC3 operator*(const VEC3& v, float t) {
    return VEC3(t * v.e[0], t * v.e[1], t * v.e[2]);
}


__host__ __device__ inline float dot(const VEC3& v1, const VEC3& v2) {
    return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

__host__ __device__ inline VEC3 cross(const VEC3& v1, const VEC3& v2) {
    return VEC3((v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]),
        (-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0])),
        (v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]));
}


__host__ __device__ inline VEC3& VEC3::operator+=(const VEC3& v) {
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
}

__host__ __device__ inline VEC3& VEC3::operator*=(const VEC3& v) {
    e[0] *= v.e[0];
    e[1] *= v.e[1];
    e[2] *= v.e[2];
    return *this;
}

__host__ __device__ inline VEC3& VEC3::operator/=(const VEC3& v) {
    e[0] /= v.e[0];
    e[1] /= v.e[1];
    e[2] /= v.e[2];
    return *this;
}

__host__ __device__ inline VEC3& VEC3::operator-=(const VEC3& v) {
    e[0] -= v.e[0];
    e[1] -= v.e[1];
    e[2] -= v.e[2];
    return *this;
}

__host__ __device__ inline VEC3& VEC3::operator*=(const float t) {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
}

__host__ __device__ inline VEC3& VEC3::operator/=(const float t) {
    float k = 1.0 / t;

    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
    return *this;
}

__host__ __device__ inline VEC3 normalize(VEC3 v) {
    return v / v.norm();
}


#endif