#include "sdf_utils.cuh"

using namespace glm;

template <typename Map> __device__ vec3 calcNormal(vec3 p, Map map) {
    vec2 e = vec2(1.0, -1.0) * 0.5773f * 0.0001f;
    return normalize(e.xyy * map(p + e.xyy) + e.yyx * map(p + e.yyx) + e.yxy * map(p + e.yxy) +
                     e.xxx * map(p + e.xxx));
}

template <typename Map> __device__ float trace_sdf(Ray r, Map map) {
    const float tmax = 16.0;
    float t = 0.01;
    for (int i = 0; i < 128; i++) {
        float h = map(r.org + r.dir * t);
        if (h < 0.0001f || t > tmax)
            break;
        t += h;
    }
    return (t < tmax) ? t : -1.0;
}
