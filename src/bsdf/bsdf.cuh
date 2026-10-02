#pragma once

#include "../scene_structs.h"
#include "../thrust_utils.h"
#include <glm/glm.hpp>

__device__ BsdfSample sample_bsdf(glm::vec3 p, glm::vec3 nor, glm::vec3 wo_w, const Material& m,
                                  RngEng& rng);

__device__ glm::vec3 eval_bsdf(glm::vec3 p, glm::vec3 nor, glm::vec3 wo_w, glm::vec3 wi_w,
                               const Material& m);

__device__ float pdf_bsdf(glm::vec3 p, glm::vec3 nor, glm::vec3 wo_w, glm::vec3 wi_w,
                          const Material& m);
