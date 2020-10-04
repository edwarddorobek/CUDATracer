#pragma once

#ifndef SCENE_H
#define SCENE_H

#include "SceneItem.h"

#include <vector>

class Scene : public SceneItem
{
public:
	__device__ Scene() {}
    __device__ Scene(SceneItem** l, int n) : m_list(l), m_size(n) {}
	__device__ virtual bool hit(const Ray& r, float tmin, float tmax, hitRecord& rec) const;
	SceneItem** m_list;
	int m_size;

};

__device__ bool Scene::hit(const Ray& r, float tmin, float tmax, hitRecord& rec) const
{
    hitRecord temp_rec;
    bool hitAnything = false;
    float closestHit = tmax;
    for (int i = 0; i < m_size; i++) {
        if (m_list[i]->hit(r, tmin, closestHit, temp_rec)) {
            hitAnything = true;
            closestHit = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hitAnything;
}

#endif // !SCENE_H
