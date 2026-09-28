#include "sdf_scene.cuh"

using namespace glm;

__device__ float scene_sdf(glm::vec3 p) {
    const float r = 1.f;

    return length(p) - r;
}
