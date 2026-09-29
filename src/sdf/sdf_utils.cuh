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
    const float EPS = 1e-5;
    const float T_MAX = 100.0;
    const int MAX_STEPS = 128;

    float t = 0.f;
    for (int i = 0; i < MAX_STEPS; i++) {
        float d = map(r.org + r.dir * t);
        if (glm::abs(d) < EPS)
            return t;
        t += glm::abs(d);

        if (t > T_MAX) {
            break;
        }
    }
    return -1.f;
}
