#include "sampling.cuh"
#include "thrust_utils.h"
#include "utilities.h"
__device__ glm::vec3 calculate_random_direction_in_cosine_hemisphere(RngEng& rng) {
    UnifDist<float> u01(0, 1);

    glm::vec2 xi(u01(rng), u01(rng));
    glm::vec3 dir = square_to_hemisphere_cosine(xi);

    return dir;
}

__device__ glm::vec2 sample_uniform_disk(RngEng& rng) {
    UnifDist<float> u01(0, 1);

    glm::vec2 xi(u01(rng), u01(rng));

    return glm::vec2(square_to_disk_concentric(xi));
}

__device__ glm::vec3 square_to_disk_concentric(glm::vec2 xi) {
    float x = xi.x * 2.f - 1.f;
    float y = xi.y * 2.f - 1.f;

    if (x == 0.f && y == 0.f) {
        return glm::vec3(0, 0, 0);
    }

    float r;
    float theta;

    if (abs(x) > abs(y)) {
        r = x;
        theta = PI / 4.f * y / x;
    } else {
        r = y;
        theta = PI / 2.f - (PI / 4.f * x / y);
    }

    return glm::vec3(r * cos(theta), r * sin(theta), 0);
}

__device__ glm::vec3 square_to_hemisphere_cosine(glm::vec2 xi) {
    glm::vec3 disk = square_to_disk_concentric(xi);
    return glm::vec3(disk.x, disk.y, sqrt(1 - disk.x * disk.x - disk.y * disk.y));
}

__device__ float square_to_hemisphere_cosine_pdf(glm::vec3 s) {
    return s.z * INV_PI;
}

__device__ glm::vec3 square_to_sphere_uniform(glm::vec2 xi) {
    float z = 1 - 2 * xi.x;
    float z_comp = sqrt(1 - z * z);
    float x = cos(2 * PI * xi.y) * z_comp;
    float y = sin(2 * PI * xi.y) * z_comp;

    return glm::vec3(x, y, z);
}

__device__ float square_to_sphere_uniform_pdf(glm::vec3 s) {
    return INV_FOUR_PI;
}
