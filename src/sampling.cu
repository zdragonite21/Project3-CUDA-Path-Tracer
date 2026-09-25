#include "sampling.cuh"
#include "thrust_utils.h"
#include "utilities.h"
__device__ glm::vec3
calculateRandomDirectionInCosineHemisphere(RngEng &rng) {
    UnifDist<float> u01(0, 1);

    glm::vec2 xi(u01(rng), u01(rng));
    glm::vec3 dir = squareToHemisphereCosine(xi);

    return dir;
}

__device__ glm::vec2 sampleUniformDisk(RngEng &rng) {
    UnifDist<float> u01(0, 1);

    glm::vec2 xi(u01(rng), u01(rng));

    return glm::vec2(squareToDiskConcentric(xi));
}

__device__ glm::vec3 squareToDiskConcentric(glm::vec2 xi) {
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

__device__ glm::vec3 squareToHemisphereCosine(glm::vec2 xi) {
    glm::vec3 disk = squareToDiskConcentric(xi);
    return glm::vec3(disk.x, disk.y,
                     sqrt(1 - disk.x * disk.x - disk.y * disk.y));
}

__device__ float squareToHemisphereCosinePDF(glm::vec3 s) {
    return s.z * INV_PI;
}

__device__ glm::vec3 squareToSphereUniform(glm::vec2 xi) {
    float z = 1 - 2 * xi.x;
    float zComp = sqrt(1 - z * z);
    float x = cos(2 * PI * xi.y) * zComp;
    float y = sin(2 * PI * xi.y) * zComp;

    return glm::vec3(x, y, z);
}

__device__ float squareToSphereUniformPDF(glm::vec3 s) { return INV_FOUR_PI; }
