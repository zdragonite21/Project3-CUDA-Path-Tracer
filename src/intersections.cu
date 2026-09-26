#include "intersections.h"
#include "scene_structs.h"
#include <cfloat>
#include <float.h>

__host__ __device__ float box_intersection_test(const Geom& box, Ray r, glm::vec3* intersectionPoint,
                                              glm::vec3* normal, bool* outside) {

    glm::vec3 ro = multiply_mv(box.transform.inverse, glm::vec4(r.org, 1.0f));
    glm::vec3 rd = multiply_mv(box.transform.inverse, glm::vec4(r.dir, 0.0f));

    float tNear = -FLT_MAX;
    float tFar = FLT_MAX;

    for (int axis = 0; axis < 3; ++axis) {
        if (rd[axis] == 0.0f) {
            if (ro[axis] < -0.5f || ro[axis] > 0.5f)
                return -1.0f;
            continue;
        }

        float t0 = (-0.5f - ro[axis]) / rd[axis];
        float t1 = (0.5f - ro[axis]) / rd[axis];
        if (t0 > t1) {
            float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }

        tNear = glm::max(tNear, t0);
        tFar = glm::min(tFar, t1);
    }

    if (tNear > tFar || tFar <= 0.0f) {
        return -1.0f;
    }

    bool out = tNear > 0.0f;
    float t = out ? tNear : tFar;
    if (outside) {
        *outside = out;
    }
    if (intersectionPoint) {
        *intersectionPoint = get_point_on_ray(r, t);
    }
    if (normal) {
        glm::vec3 p = ro + t * rd;
        glm::vec3 a = glm::abs(p);

        glm::vec3 nor(0.0f);
        if (a.x > a.y && a.x > a.z)
            nor.x = glm::sign(p.x);
        else if (a.y > a.z)
            nor.y = glm::sign(p.y);
        else
            nor.z = glm::sign(p.z);
        *normal = glm::normalize(multiply_mv(box.transform.inv_transpose, glm::vec4(nor, 0.0f)));
    }

    return t;
}

__host__ __device__ float sphere_intersection_test(const Geom& sphere, Ray r,
                                                 glm::vec3* intersectionPoint, glm::vec3* normal,
                                                 bool* outside) {
    glm::vec3 ro = multiply_mv(sphere.transform.inverse, glm::vec4(r.org, 1.0f));
    glm::vec3 rd = multiply_mv(sphere.transform.inverse, glm::vec4(r.dir, 0.0f));

    float a = glm::dot(rd, rd);
    float half_b = glm::dot(ro, rd);
    float c = glm::dot(ro, ro) - 1.0f;

    float disc = half_b * half_b - a * c;
    if (disc < 0.f) {
        return -1.f;
    }

    float sqrtD = glm::sqrt(disc);

    float t = (-half_b - sqrtD) / a;

    constexpr float eps = 1e-4f;
    if (t <= eps) {
        t = (-half_b + sqrtD) / a;
        if (t <= eps) {
            return -1.f;
        }
    }

    if (outside) {
        *outside = dot(ro, ro) >= 1.f;
    }
    if (intersectionPoint) {
        *intersectionPoint = get_point_on_ray(r, t);
    }
    if (normal) {
        glm::vec3 p = ro + rd * t;
        *normal = glm::normalize(multiply_mv(sphere.transform.inv_transpose, glm::vec4(p, 0.f)));
    }

    return t;
}

__host__ __device__ float plane_intersection_test(const Geom& plane, Ray r,
                                                glm::vec3* intersectionPoint, glm::vec3* normal,
                                                bool* backFacing) {

    glm::vec3 ro = multiply_mv(plane.transform.inverse, glm::vec4(r.org, 1.0f));
    glm::vec3 rd = multiply_mv(plane.transform.inverse, glm::vec4(r.dir, 0.0f));

    if (glm::abs(rd.y) <= 0.0001) {
        return -1;
    }

    float t = -ro.y / rd.y;
    if (t <= 0.0001) {
        return -1;
    }

    glm::vec3 pW = ro + rd * t;

    if (glm::abs(pW.x) > 0.5 || glm::abs(pW.z) > 0.5) {
        return -1;
    }

    if (backFacing) {
        *backFacing = rd.y > 0;
    }
    if (intersectionPoint) {
        *intersectionPoint = get_point_on_ray(r, t);
    }
    if (normal) {
        *normal = glm::normalize(multiply_mv(plane.transform.inv_transpose, glm::vec4(0, 1, 0, 0)));
    }
    return t;
}

__device__ bool visible_to_light(Ray r, int lightGeomIdx, float lightDist, const Geom* geoms, int geoms_size) {
    float minT = lightDist;
    float t;

    for (int i = 0; i < geoms_size; ++i) {
        if (i == lightGeomIdx) {
            continue;
        }
        const Geom& geom = geoms[i];
        switch (geom.type) {
        case GeomType::Cube:
            t = box_intersection_test(geom, r, nullptr, nullptr, nullptr);
            break;
        case GeomType::Plane:
            t = plane_intersection_test(geom, r, nullptr, nullptr, nullptr);
            break;
        case GeomType::Sphere:
            t = sphere_intersection_test(geom, r, nullptr, nullptr, nullptr);
            break;
        }
        if (t > 0 && t < minT) {
            return false;
        }
    }
    return true;
}
