#pragma once
#include "sceneStructs.h"
#include "thrust_utils.h"
#include <cuda_runtime.h>
#include <cuda/std/optional>

namespace cstd = cuda::std;

__device__ cstd::optional<LightSample> directSamplePlaneLight(glm::vec3 p, const Geom& plane, RngEng& rng);
__device__ cstd::optional<LightSample> directSampleAreaLight(glm::vec3 p, const Geom& geom, RngEng& rng);
__device__ cstd::optional<LightSample> sampleLi(glm::vec3 p, glm::vec3 nor, const Light* lights, int lights_size,
                                const Geom* geoms, int geoms_size, RngEng& rng);
