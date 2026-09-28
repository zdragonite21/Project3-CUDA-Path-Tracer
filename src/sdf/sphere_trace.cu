#include "../scene_structs.h"
#include "sdf_scene.cuh"
#include <cuda_runtime.h>
#include <glm/glm.hpp>


using namespace glm;

__device__ float trace_sdf(Ray r) {
    // sphere march scene_sdf
    const float tmax = 16.0;
    float t = 0.01;
    for (int i = 0; i < 128; i++) {
        float h = scene_sdf(r.org + r.dir * t);
        if (h < 0.0001f || t > tmax)
            break;
        t += h;
    }
    return (t < tmax) ? t : -1.0;
}
