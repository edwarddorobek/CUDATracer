#pragma once

#ifndef MATERIAL_H
#define MATERIAL_H

struct hitRecord;

#include "CUDA_RT.h"
#include "SceneItem.h"

__device__ float schlickApprox(float cosine, float refIdx)
{
	float r0 = (1.0f - refIdx) / (1.0f + refIdx);
	r0 = r0 * r0;
	return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

__device__ bool refract(const VEC3& v, const VEC3& n, float ni_nt, VEC3& refracted) {
    VEC3 uv = normalize(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_nt * ni_nt * (1 - dt * dt);
    if (discriminant > 0) {
        refracted = ni_nt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    }
    else
        return false;
}

__device__ VEC3 reflect(const VEC3& v, const VEC3& n) {
    return v - 2.0f * dot(v, n) * n;
}


class Material 
{
public:
    __device__ virtual VEC3 emitted(const VEC3& p) const { return VEC3(0,0,0); }
    __device__ virtual bool scatter(const Ray& r_in, const hitRecord& rec, VEC3& attenuation, Ray& scattered, curandState* local_rand_state) const = 0;
};

class Lambertian : public Material 
{
public:
    __device__ Lambertian(const VEC3& a) : m_albedo(a) {}
    __device__ virtual bool scatter(const Ray& r_in, const hitRecord& rec, VEC3& attenuation, Ray& scattered, curandState* local_rand_state) const 
    {
        VEC3 target = rec.p + rec.normal + randVecSphere(local_rand_state);
        scattered = Ray(rec.p, target - rec.p);
        attenuation = m_albedo;
        return true;
    }

    VEC3 m_albedo;
};


class Metal : public Material 
{
public:
    __device__ Metal(const VEC3& a, float f) : m_albedo(a) { if (f < 1) m_fuzz = f; else m_fuzz = 1; }
    __device__ virtual bool scatter(const Ray& r_in, const hitRecord& rec, VEC3& attenuation, Ray& scattered, curandState* local_rand_state) const 
    {
        VEC3 reflected = reflect(normalize(r_in.direction()), rec.normal);
        scattered = Ray(rec.p, reflected + m_fuzz * randVecSphere(local_rand_state));
        attenuation = m_albedo;
        return (dot(scattered.direction(), rec.normal) > 0.0f);
    }
    VEC3 m_albedo;
    float m_fuzz;
};


class Dielectric : public Material {
public:
    __device__ Dielectric(float ri) : m_refIdx(ri) {}
    __device__ virtual bool scatter(const Ray& r_in,
        const hitRecord& rec,
        VEC3& attenuation,
        Ray& scattered,
        curandState* local_rand_state) const 
    {
        VEC3 outward_normal;
        VEC3 reflected = reflect(r_in.direction(), rec.normal);
        float ni_nt;

        attenuation = VEC3(1.0, 1.0, 1.0);

        VEC3 refracted;
        float reflect_prob;
        float cosine;

        if (dot(r_in.direction(), rec.normal) > 0.0f) {
            outward_normal = -rec.normal;
            ni_nt = m_refIdx;
            cosine = dot(r_in.direction(), rec.normal) / r_in.direction().norm();
            cosine = sqrt(1.0f - m_refIdx * m_refIdx * (1 - cosine * cosine));
        }
        else {
            outward_normal = rec.normal;
            ni_nt = 1.0f / m_refIdx;
            cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().norm();
        }
        if (refract(r_in.direction(), outward_normal, ni_nt, refracted))
            reflect_prob = schlickApprox(cosine, m_refIdx);
        else
            reflect_prob = 1.0f;
        if (curand_uniform(local_rand_state) < reflect_prob)
            scattered = Ray(rec.p, reflected);
        else
            scattered = Ray(rec.p, refracted);
        return true;
    }

    float m_refIdx;
};

class DiffuseLight : public Material
{
public:
    __device__ DiffuseLight(const VEC3& a) : m_emit(a) {}

    __device__ virtual bool scatter(const Ray& r_in,
        const hitRecord& rec,
        VEC3& attenuation,
        Ray& scattered,
        curandState* local_rand_state) const
    {
        return false;
    }

    __device__ virtual VEC3 emitted(const VEC3& p) const
    {
        return m_emit;
    }

    VEC3 m_emit;
};

class Hybrid : public Material
{
public:
    __device__ Hybrid(const VEC3& a, float ints) : m_emit(a), m_intensity(ints) {}

    __device__ virtual bool scatter(const Ray& r_in,
            const hitRecord& rec,
            VEC3& attenuation,
            Ray& scattered,
            curandState* local_rand_state) const
    {
        VEC3 target = rec.p + rec.normal + randVecSphere(local_rand_state);
        scattered = Ray(rec.p, target - rec.p);
        attenuation = m_emit;
        return true;
    }

    __device__ virtual VEC3 emitted(const VEC3& p) const
    {
        return m_emit * m_intensity;
    }

    VEC3 m_emit;
    float m_intensity;
};


#endif