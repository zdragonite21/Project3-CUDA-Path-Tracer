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
    float dz = 1.0;

    for (int i = 0; i < 3; i++) {
        // trigonometric version (MUCH faster than polynomial)

        // dz = 8*z^7*dz
        dz = 8.0 * pow(m, 3.5) * dz + 1.0;

        // z = z^8+c
        float r = length(w);
        float b = 8.0 * acos(w.y / r);
        float a = 8.0 * atan(w.x, w.z);
        w = p + powf(r, 8.0) * vec3(sin(b) * sin(a), cos(b), sin(b) * cos(a));

        // trap = min(trap, vec4(abs(w), m));

        m = dot(w, w);
        if (m > 256.0)
            break;
    }

    // resColor = vec4(m, trap.yzw);

    // distance estimation (through the Hubbard-Douady potential)
    return 0.25 * log(m) * sqrt(m) / dz;
}
