#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>


__device__ float scene_sdf(glm::vec3 p);
