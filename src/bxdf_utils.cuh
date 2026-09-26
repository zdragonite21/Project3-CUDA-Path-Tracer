#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include "scene_structs.h"

// bxdf utils
namespace bx {
static __device__ __forceinline__ float cos_theta(const glm::vec3 &w) {
    return w.z;
}
static __device__ __forceinline__ float cos2_theta(const glm::vec3 &w) {
    return w.z * w.z;
}
static __device__ __forceinline__ float sin2_theta(const glm::vec3 &w) {
    return glm::max(0.0, 1.0 - cos2_theta(w));
}
static __device__ __forceinline__ float sin_theta(const glm::vec3 &w) {
    return glm::sqrt(sin2_theta(w));
}
static __device__ __forceinline__ float tan_theta(const glm::vec3 &w) {
    return sin_theta(w) / cos_theta(w);
}
static __device__ __forceinline__ float tan2_theta(const glm::vec3 &w) {
    return sin2_theta(w) / cos2_theta(w);
}
static __device__ __forceinline__ float cos_phi(const glm::vec3 &w) {
    float sinTheta = sin_theta(w);
    return (sinTheta == 0) ? 0 : glm::clamp(w.x / sinTheta, -1.f, 1.f);
}
static __device__ __forceinline__ float sin_phi(const glm::vec3 &w) {
    float sinTheta = sin_theta(w);
    return (sinTheta == 0) ? 0 : glm::clamp(w.y / sinTheta, -1.f, 1.f);
}
static __device__ __forceinline__ float cos2_phi(const glm::vec3 &w) {
    return cos_phi(w) * cos_phi(w);
}
static __device__ __forceinline__ float sin2_phi(const glm::vec3 &w) {
    return sin_phi(w) * sin_phi(w);
}
// difference of azimuth angles between two vectors
static __device__ __forceinline__ float cos_d_phi(const glm::vec3 &wa,
                                                const glm::vec3 &wb) {
    return glm::clamp((wa.x * wb.x + wa.y * wb.y) /
                          glm::sqrt((wa.x * wa.x + wa.y * wa.y) *
                                    (wb.x * wb.x + wb.y * wb.y)),
                      -1.f, 1.f);
}
static __device__ __forceinline__ glm::vec3 Faceforward(const glm::vec3 &n,
                                                const glm::vec3 &v) {
    return glm::dot(n, v) < 0.f ? -n : n;
}
static __device__ void coordinate_system(glm::vec3 in_nor, glm::vec3 &out_tan,
                                        glm::vec3 &out_bit) {
    if (abs(in_nor.x) > abs(in_nor.y))
        out_tan = glm::vec3(-in_nor.z, 0, in_nor.x) /
                  sqrt(in_nor.x * in_nor.x + in_nor.z * in_nor.z);
    else
        out_tan = glm::vec3(0, in_nor.z, -in_nor.y) /
                  sqrt(in_nor.y * in_nor.y + in_nor.z * in_nor.z);
    out_bit = glm::cross(in_nor, out_tan);
}

static __device__ glm::mat3 local_to_world(glm::vec3 nor) {
    glm::vec3 tan, bit;
    coordinate_system(nor, tan, bit);
    return glm::mat3(tan, bit, nor);
}

static __device__ glm::mat3 world_to_local(glm::vec3 nor) {
    return glm::transpose(local_to_world(nor));
}

static __device__ bool Refract(glm::vec3 wo, glm::vec3 n, float eta, glm::vec3 &wt) {
    // Compute cos theta using Snell's law
    float cosThetaI = glm::dot(n, wo);
    float sin2ThetaI = glm::max(0.f, 1.f - cosThetaI * cosThetaI);
    float sin2ThetaT = eta * eta * sin2ThetaI;

    // Handle total internal reflection for transmission
    if (sin2ThetaT >= 1.f)
        return false;
    float cosThetaT = glm::sqrt(1.f - sin2ThetaT);
    wt = eta * -wo + (eta * cosThetaI - cosThetaT) * n;
    return true;
}
static __device__ Ray spawn_ray(glm::vec3 pos, glm::vec3 wi) {
    return Ray{pos + wi * 0.0001f, wi};
}
} // namespace bx
