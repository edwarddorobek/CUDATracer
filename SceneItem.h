#pragma once

#ifndef SCENEITEM_H
#define SCENEITEM_H

#include "CUDA_RT.h"
#include "Ray.h"

class Material;

struct hitRecord
{
	float t;
	VEC3 p;
	VEC3 normal;
	bool frontFace;
	Material* matPtr;

	__device__ inline void setFaceNormal(const Ray& r, const VEC3& outwardNormal)
	{
		frontFace = dot(r.direction(), outwardNormal) < 0;
		normal = frontFace ? outwardNormal : -outwardNormal;
	}
};

class SceneItem 
{
public:
	__device__ virtual bool hit(const Ray& r, float tMin, float tMax, hitRecord& rec) const = 0;
};

#endif 
