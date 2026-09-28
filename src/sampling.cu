#include "sampling.cuh"
#include "utilities.h"

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

__host__ __device__ RngEng make_seeded_rng(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return RngEng(h);
}

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
    return glm::vec3(disk.x, disk.y, sqrt(glm::max(1 - disk.x * disk.x - disk.y * disk.y, 0.f)));
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
