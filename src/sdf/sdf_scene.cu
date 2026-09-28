#include "sdf_scene.cuh"
#include "sdf_utils.cuh"
#include "sdf_structs.cuh"

using namespace glm;

__device__ float scene_intersect(Ray r) {
    return trace_sdf(r, SphereSDF{});
}

__device__ vec3 scene_normal(vec3 p) {
    return calc_normal(p, SphereSDF{});
}
