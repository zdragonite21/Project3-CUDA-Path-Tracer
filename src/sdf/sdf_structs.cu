#include "sdf_structs.cuh"

using namespace glm;

__device__ float SphereSDF::operator()(vec3 p) const {
    const float r = 1.f;
    return length(p) - r;
}

__device__ float MandelbulbSDF::operator()(vec3 p) const {
    vec3 w = p;
    float m = dot(w, w);

    // vec4 trap = vec4(abs(w), m);
    float dz = 1.f;

    for (int i = 0; i < 3; ++i) {
        // trigonometric version (MUCH faster than polynomial)

        // dz = 8*z^7*dz
        dz = 8.f * __powf(m, 3.5f) * dz + 1.f;

        // z = z^8+c
        float r = length(w);
        float b = 8.f * acosf(w.y / r);
        float a = 8.f * atan(w.x, w.z);
        w = p + __powf(r, 8.f) * vec3(__sinf(b) * __sinf(a), __cosf(b), __sinf(b) * __cosf(a));

        // trap = min(trap, vec4(abs(w), m));

        m = dot(w, w);
        if (m > 256.f)
            break;
    }

    // resColor = vec4(m, trap.yzw);

    // distance estimation (through the Hubbard-Douady potential)
    return 0.25f * __logf(m) * sqrtf(m) / dz;
}
