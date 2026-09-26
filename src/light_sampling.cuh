#pragma once
#include "scene_structs.h"
#include "thrust_utils.h"
#include <cuda_runtime.h>
#include <cuda/std/optional>

namespace cstd = cuda::std;

__device__ cstd::optional<LightSample> sample_plane_light(glm::vec3 p, const Geom& plane, RngEng& rng);
__device__ cstd::optional<LightSample> sample_area_light(glm::vec3 p, const Geom& geom, RngEng& rng);
__device__ cstd::optional<LightSample> sample_li(glm::vec3 p, glm::vec3 nor, const Light* lights, int num_lights,
                                const Geom* geoms, int num_geoms, RngEng& rng);
