#pragma once

#ifndef PRIMITIVES_H
#define PRIMITIVES_H

#include "CUDA_RT.h"
#include "SceneItem.h"
#include <math.h>


/* ------------------------ Sphere Class ------------------------ */
__device__ void getSphereUV(const VEC3& p, float& u, float& v) {
    float phi = atan2(p.z(), p.x());
    float theta = asin(p.y());
    u = 1 - (phi + C_PI()) / (2 * C_PI());
    v = (theta + C_PI() / 2) / C_PI();
}

class Sphere : public SceneItem
{
public:
	__device__ Sphere() {}
	__device__ Sphere(VEC3 center, float rad, Material* mat) : m_center(center), m_radius(rad), m_matPtr(mat) {}

	__device__ virtual bool hit(const Ray& r, float tmin, float tmax, hitRecord& rec) const;

	VEC3 m_center;
	float m_radius;
    Material* m_matPtr;
};

__device__ bool Sphere::hit(const Ray& r, float tmin, float tmax, hitRecord& rec) const
{
    VEC3 oc = r.origin() - m_center;
    auto a = r.direction().squaredNorm();
    auto half_b = dot(oc, r.direction());
    auto c = oc.squaredNorm() - m_radius * m_radius;
    auto discriminant = half_b * half_b - a * c;

    if (discriminant > 0) {
        auto root = sqrt(discriminant);
        auto temp = (-half_b - root) / a;
        if (temp < tmax && temp > tmin) {
            rec.t = temp;
            rec.p = r.at(rec.t);
            VEC3 normal = (rec.p - m_center) / m_radius;
            rec.setFaceNormal(r, normal);
            rec.matPtr = m_matPtr;
            return true;
        }
        temp = (-half_b + root) / a;
        if (temp < tmax && temp > tmin) {
            rec.t = temp;
            rec.p = r.at(rec.t);
            VEC3 normal = (rec.p - m_center) / m_radius;
            rec.setFaceNormal(r, normal);
            rec.matPtr = m_matPtr;
            return true;
        }
    }
    return false;
}

/* ------------------------ Triangle Class ------------------------ */

#endif 
