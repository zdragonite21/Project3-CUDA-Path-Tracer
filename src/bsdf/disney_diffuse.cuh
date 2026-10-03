#pragma once
#include "../scene_structs.h"
#include "../thrust_utils.h"

__device__ __forceinline__ glm::vec3 disney_diffuse_eval(glm::vec3 wo, glm::vec3 wi,
                                                         glm::vec3 base_color, float roughness,
                                                         float subsurface);
__device__ __forceinline__ float disney_diffuse_pdf(glm::vec3 wo, glm::vec3 wi);
__device__ __forceinline__ BsdfSample disney_diffuse_sample(glm::vec3 wo, glm::vec3 base_color,
                                                            float roughness, float subsurface,
                                                            RngEng& rng);
