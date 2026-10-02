#pragma once

#include "thrust_utils.h"
#include <glm/glm.hpp>

__host__ __device__ RngEng make_seeded_rng(int iter, int index, int depth);
__device__ glm::vec3 calculate_random_direction_in_cosine_hemisphere(RngEng& rng);
__device__ glm::vec2 sample_uniform_disk(RngEng& rng);
__device__ glm::vec3 sample_uniform_sphere_cap(RngEng& rng, float zmin);
__device__ glm::vec3 square_to_disk_concentric(glm::vec2 xi);
__device__ glm::vec3 square_to_hemisphere_cosine(glm::vec2 xi);
__device__ float square_to_hemisphere_cosine_pdf(glm::vec3 s);
__device__ glm::vec3 square_to_sphere_uniform(glm::vec2 xi);
__device__ float square_to_sphere_uniform_pdf(glm::vec3 s);
__device__ glm::vec3 square_to_sphere_cap(glm::vec2 xi, float zmin);
