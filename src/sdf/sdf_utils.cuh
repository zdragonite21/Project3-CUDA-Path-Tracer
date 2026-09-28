#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "../scene_structs.h"

template <typename Map> __device__ glm::vec3 calc_normal(glm::vec3 p, Map map);

template <typename Map> __device__ float trace_sdf(Ray r, Map map);
