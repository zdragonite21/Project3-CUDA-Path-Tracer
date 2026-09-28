#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>

struct SphereSDF {
    __device__ float operator()(glm::vec3 p) const;
};
