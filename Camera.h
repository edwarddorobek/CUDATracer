#pragma once

#ifndef CAMERA_H
#define CAMERA_H

#include "CUDA_RT.h"

class Camera
{
public:
	__device__ Camera(VEC3 lookFrom, VEC3 lookAt, VEC3 vup, float vfov, 
		float aspect, float aperture, float focusDist) 
	{
		m_lensRadius = aperture / 2.f;
		float theta = vfov * (C_PI()) / 180.f;
		float halfHeight = tan(theta / 2.f);
		float halfWidth = aspect * halfHeight;
		m_origin = lookFrom;
		m_w = normalize(lookFrom - lookAt);
		m_u = normalize(cross(vup, m_w));
		m_v = cross(m_w, m_u);
		m_lowerLeft = m_origin - halfWidth * focusDist * m_u - halfHeight * focusDist * m_v - focusDist * m_w;
		m_horizontal = 2.f * halfWidth * focusDist * m_u;
		m_vertical = 2.f * halfHeight * focusDist * m_v;
	}

	__device__ Ray generateRay(float s, float t, curandState* threadRandState) 
	{ 
		VEC3 rd = m_lensRadius * randVecDisk(threadRandState);
		VEC3 offset = m_u * rd.x() + m_v * rd.y();
		return Ray(m_origin + offset, m_lowerLeft + s * m_horizontal + t * m_vertical - m_origin - offset);
	}

	VEC3 m_origin;
	VEC3 m_lowerLeft;
	VEC3 m_horizontal;
	VEC3 m_vertical;
	VEC3 m_u, m_v, m_w;
	float m_lensRadius;
};

#endif