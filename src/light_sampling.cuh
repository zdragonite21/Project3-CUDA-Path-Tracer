#pragma once
#include "scene_structs.h"
#include "thrust_utils.h"
#include <cuda/std/optional>
#include <cuda_runtime.h>

namespace cstd = cuda::std;

__device__ cstd::optional<LightSample> sample_direct_light(glm::vec3 p, glm::vec3 nor,
                                                           const Light* lights, int num_lights,
                                                           const Geom* geoms, int num_geoms,
                                                           RngEng& rng);

__device__ float pdf_li(const Ray& ray, const Light& light, const Geom* geoms, int num_geoms);
