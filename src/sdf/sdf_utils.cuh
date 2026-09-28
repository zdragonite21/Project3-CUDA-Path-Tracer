#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "../scene_structs.h"

template <typename Map> __device__ glm::vec3 calc_normal(glm::vec3 p, Map map) {
    const float e = 0.5773f * 0.0001f;
    const glm::vec3 k1(e, -e, -e);
    const glm::vec3 k2(-e, -e, e);
    const glm::vec3 k3(-e, e, -e);
    const glm::vec3 k4(e, e, e);
    return glm::normalize(k1 * map(p + k1) + k2 * map(p + k2) + k3 * map(p + k3) +
                          k4 * map(p + k4));
}

template <typename Map> __device__ float trace_sdf(Ray r, Map map) {
    const float tmax = 16.0;
    float t = 0.01;
    for (int i = 0; i < 128; i++) {
        float h = map(r.org + r.dir * t);
        if (h < 0.0001f || t > tmax)
            break;
        t += h;
    }
    return (t < tmax) ? t : -1.0;
}
