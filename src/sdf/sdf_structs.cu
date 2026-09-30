#include "sdf_structs.cuh"
#include <glm/geometric.hpp>

using namespace glm;

__device__ float SphereSDF::operator()(vec3 p) const {
    const float r = 1.f;
    return length(p) - r;
}

__device__ inline void double_angle(float& s, float& c) {
    float s2 = 2.f * s * c;
    float c2 = c * c - s * s;

    s = s2;
    c = c2;
}

__device__ __forceinline__ glm::vec3 calc_w(glm::vec3 w, const glm::vec3& p) {
    float x = w.x;
    float y = w.y;
    float z = w.z;

    float r2 = x * x + y * y + z * z;
    if (r2 <= 1e-20f)
        return p;

    float inv_r = rsqrtf(r2);

    // theta = acos(y / r)
    float cos_theta = y * inv_r;
    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));

    // phi = atan2(x, z)
    float rho2 = x * x + z * z;

    float sin_phi;
    float cos_phi;
    if (rho2 > 0.00001f) {
        float inv_rho = rsqrtf(rho2);
        sin_phi = x * inv_rho;
        cos_phi = z * inv_rho;
    } else {
        // undefined at the pole
        sin_phi = 0.0f;
        cos_phi = 1.0f;
    }

    // 8theta
    double_angle(sin_theta, cos_theta);
    double_angle(sin_theta, cos_theta);
    double_angle(sin_theta, cos_theta);

    // 8phi
    double_angle(sin_phi, cos_phi);
    double_angle(sin_phi, cos_phi);
    double_angle(sin_phi, cos_phi);

    float r4 = r2 * r2;
    float r8 = r4 * r4;

    return p + r8 * glm::vec3(sin_theta * sin_phi, cos_theta, sin_theta * cos_phi);
}

__device__ float MandelbulbDE::operator()(vec3 p) const {
    vec3 w = p;
    float m = dot(w, w);

    // vec4 trap = vec4(abs(w), m);
    float dz = 1.f;

    for (int i = 0; i < 3; ++i) {
        // trigonometric version (MUCH faster than polynomial)

        // dz = 8*z^7*dz
        dz = 8.f * __powf(m, 3.5f) * dz + 1.f;

        // z = z^8+c
        w = calc_w(w, p);

        // trap = min(trap, vec4(abs(w), m));

        m = dot(w, w);
        if (m > 256.f)
            break;
    }

    // resColor = vec4(m, trap.yzw);

    // distance estimation (through the Hubbard-Douady potential)
    return 0.25f * __logf(m) * sqrtf(m) / dz;
}
