#include "sampling.cuh"
#include "utilities.h"

__device__ void coordinateSystem(glm::vec3 in_nor, glm::vec3 &out_tan,
                                 glm::vec3 &out_bit) {
    if (abs(in_nor.x) > abs(in_nor.y))
        out_tan = glm::vec3(-in_nor.z, 0, in_nor.x) /
                  sqrt(in_nor.x * in_nor.x + in_nor.z * in_nor.z);
    else
        out_tan = glm::vec3(0, in_nor.z, -in_nor.y) /
                  sqrt(in_nor.y * in_nor.y + in_nor.z * in_nor.z);
    out_bit = glm::cross(in_nor, out_tan);
}

__device__ glm::vec3
calculateRandomDirectionInCosineHemisphere(glm::vec3 normal,
                                           thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    glm::vec2 xi(u01(rng), u01(rng));
    glm::vec3 dir = squareToHemisphereCosine(xi);

    glm::vec3 tan;
    glm::vec3 bit;
    coordinateSystem(normal, tan, bit);
    return dir.z * normal + dir.y * tan + dir.x * bit;
}

__device__ glm::vec2 sampleUniformDisk(thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

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
