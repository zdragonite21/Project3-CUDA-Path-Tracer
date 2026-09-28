#include "sdf_structs.cuh"

using namespace glm;

__device__ float SphereSDF::operator()(vec3 p) const {
    const float r = 1.f;
    return length(p) - r;
}
