#include "sceneStructs.h"
#include "thrust_utils.h"
#include <thrust/binary_search.h>
#include <thrust/partition.h>

struct IsPathAlive {
    __host__ __device__ bool operator()(const PathSegment& ps) const {
        return ps.remainingBounces > 0;
    }
};

struct IsIsectHit {
    template <typename Tuple>
    __device__ bool operator()(const Tuple& t) const {
        const MatId matId = thrust::get<2>(t);
        return matId < UINT8_MAX;
    }
};

void sort_paths(int num_paths, ShadeableIntersection* isects, MatId* matIds, PathSegment* paths) {
    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(isects, paths));
    thrust::sort_by_key(thrust::device, matIds, matIds + num_paths, zip_begin);
}

int filter_missed(int num_paths, MatId* matIds) {
    auto new_end = thrust::lower_bound(thrust::device, matIds, matIds + num_paths, UINT8_MAX);
    return new_end - matIds;
}

int compact_terminated(int num_paths, PathSegment* paths) {
    auto new_end = thrust::partition(thrust::device, paths, paths + num_paths, IsPathAlive{});
    return new_end - paths;
}

int compact_missed(int num_paths, ShadeableIntersection* isects, MatId* matIds,
                       PathSegment* paths) {
    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(isects, paths, matIds));
    auto new_end =
        thrust::partition(thrust::device, zip_begin, zip_begin + num_paths, IsIsectHit{});
    return new_end - zip_begin;
}
