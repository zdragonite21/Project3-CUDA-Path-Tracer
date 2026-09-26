#pragma once

#include "scene_structs.h"

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

__host__ __device__ inline glm::vec3 get_point_on_ray(Ray r, float t) {
    return r.org + t * r.dir;
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ inline glm::vec3 multiply_mv(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param isect_point  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float box_intersection_test(const Geom& box, Ray r, glm::vec3* isect_point,
                                              glm::vec3* normal, bool* outside);

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 1 and is centered at the origin.
 *
 * @param isect_point  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphere_intersection_test(const Geom& sphere, Ray r, glm::vec3* isect_point,
                                                 glm::vec3* normal, bool* outside);

__host__ __device__ float plane_intersection_test(const Geom& plane, Ray r, glm::vec3* isect_point,
                                                glm::vec3* normal, bool* outside);

__device__ bool visible_to_light(Ray r, int light_geom_idx, float light_dist, const Geom* geoms, int num_geoms);
