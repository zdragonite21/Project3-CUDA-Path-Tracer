#include <cuda_runtime.h>
#include <glm/glm.hpp>

// bxdf utils
namespace bx {
static __device__ __forceinline__ float CosTheta(const glm::vec3 &w) {
    return w.z;
}
static __device__ __forceinline__ float Cos2Theta(const glm::vec3 &w) {
    return w.z * w.z;
}
static __device__ __forceinline__ float Sin2Theta(const glm::vec3 &w) {
    return glm::max(0.0, 1.0 - Cos2Theta(w));
}
static __device__ __forceinline__ float SinTheta(const glm::vec3 &w) {
    return glm::sqrt(Sin2Theta(w));
}
static __device__ __forceinline__ float TanTheta(const glm::vec3 &w) {
    return SinTheta(w) / CosTheta(w);
}
static __device__ __forceinline__ float Tan2Theta(const glm::vec3 &w) {
    return Sin2Theta(w) / Cos2Theta(w);
}
static __device__ __forceinline__ float CosPhi(const glm::vec3 &w) {
    float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 0 : glm::clamp(w.x / sinTheta, -1.f, 1.f);
}
static __device__ __forceinline__ float SinPhi(const glm::vec3 &w) {
    float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 0 : glm::clamp(w.y / sinTheta, -1.f, 1.f);
}
static __device__ __forceinline__ float Cos2Phi(const glm::vec3 &w) {
    return CosPhi(w) * CosPhi(w);
}
static __device__ __forceinline__ float Sin2Phi(const glm::vec3 &w) {
    return SinPhi(w) * SinPhi(w);
}
// difference of azimuth angles between two vectors
static __device__ __forceinline__ float CosDPhi(const glm::vec3 &wa,
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
static __device__ void coordinateSystem(glm::vec3 in_nor, glm::vec3 &out_tan,
                                        glm::vec3 &out_bit) {
    if (abs(in_nor.x) > abs(in_nor.y))
        out_tan = glm::vec3(-in_nor.z, 0, in_nor.x) /
                  sqrt(in_nor.x * in_nor.x + in_nor.z * in_nor.z);
    else
        out_tan = glm::vec3(0, in_nor.z, -in_nor.y) /
                  sqrt(in_nor.y * in_nor.y + in_nor.z * in_nor.z);
    out_bit = glm::cross(in_nor, out_tan);
}

static __device__ glm::mat3 localToWorld(glm::vec3 nor) {
    glm::vec3 tan, bit;
    coordinateSystem(nor, tan, bit);
    return glm::mat3(tan, bit, nor);
}

static __device__ glm::mat3 worldToLocal(glm::vec3 nor) {
    return glm::transpose(localToWorld(nor));
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
} // namespace bx
