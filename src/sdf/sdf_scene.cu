#include "sdf_scene.cuh"
#include "sdf_utils.cuh"
#include "sdf_structs.cuh"

using namespace glm;

__device__ float scene_intersect(Ray r, float t_min, float t_max, float eps) {
    return trace_sdf<SphereSDF>(r, SphereSDF{}, t_min, t_max, eps);
}

__device__ vec3 scene_normal(vec3 p, float eps) {
    return calc_normal<SphereSDF>(p, SphereSDF{}, eps);
}
