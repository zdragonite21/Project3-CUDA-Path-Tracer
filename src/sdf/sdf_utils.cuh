#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "../config.h"
#include "../scene_structs.h"


template <typename Map> __device__ glm::vec3 calc_normal(glm::vec3 p, Map map, float eps) {
    const float e = 0.5773f * eps;
    const glm::vec3 k1(e, -e, -e);
    const glm::vec3 k2(-e, -e, e);
    const glm::vec3 k3(-e, e, -e);
    const glm::vec3 k4(e, e, e);
    return glm::normalize(k1 * map(p + k1) + k2 * map(p + k2) + k3 * map(p + k3) +
                          k4 * map(p + k4));
}

template <typename Map>
__device__ float trace_sdf(Ray r, Map map, float t_min, float t_max, float eps) {
    float t = t_min;
    if (t > t_max) {
        return -1.f;
    }

    for (int i = 0; i < scene_params::max_steps; i++) {
        float d = map(r.org + r.dir * t);
        if (glm::abs(d) < eps)
            return t;
        t += glm::abs(d);

        if (t > t_max) {
            break;
        }
    }
    return -1.f;
}
