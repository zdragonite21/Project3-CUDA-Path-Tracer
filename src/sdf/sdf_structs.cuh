#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>

__device__ inline void double_angle(float& s, float& c)
{
    float s2 = 2.f * s * c;
    float c2 = c*c - s*s;

    s = s2;
    c = c2;
}

struct SphereSDF {
    __device__ float operator()(glm::vec3 p) const;
};

struct MandelbulbSDF {
    __device__ float operator()(glm::vec3 p) const;
};
