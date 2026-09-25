#include "sceneStructs.h"
#include "thrust_utils.h"
#include <thrust/binary_search.h>
#include <thrust/partition.h>
#include <thrust/execution_policy.h>

struct IsPathTerminated {
    __host__ __device__ bool operator()(const PathSegment& ps) const {
        return ps.remainingBounces <= 0;
    }
};

struct IsIsectHit {
    template <typename Tuple>
    __device__ bool operator()(const Tuple& t) const {
        const MatId matId = thrust::get<2>(t);
        return matId < UINT8_MAX;
    }
};

void sort_paths(int num_paths, ShadeableIntersection* isects, MatId* matIds, PathSegment* paths, cudaStream_t stream) {
    auto policy = thrust::cuda::par_nosync.on(stream);
    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(isects, paths));
    thrust::sort_by_key(policy, matIds, matIds + num_paths, zip_begin);
}

int filter_missed(int num_paths, MatId* matIds, cudaStream_t stream) {
    auto policy = thrust::cuda::par_nosync.on(stream);
    auto new_end = thrust::lower_bound(policy, matIds, matIds + num_paths, UINT8_MAX);
    return new_end - matIds;
}

int compact_terminated(int num_paths, PathSegment* paths, cudaStream_t stream) {
    auto policy = thrust::cuda::par_nosync.on(stream);
    auto new_end = thrust::remove_if(policy, paths, paths + num_paths, IsPathTerminated{});
    return new_end - paths;
}
