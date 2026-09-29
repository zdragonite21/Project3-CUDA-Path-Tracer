#pragma once

#define ERRORCHECK 0

#define SORT_PATHS 0
#define COMPACT_TERMINATED 1

#define LI_MIS 1
#define RUSSIAN_ROULETTE 1

#define AGX_TONEMAP 1

namespace numeric {
constexpr float ray_offset = 1e-4f;
constexpr float shadow_margin = 1e-3f;
constexpr float sdf_hit = 1e-5f;
constexpr float sdf_normal = 1e-4f;
constexpr float min_cos = 1e-4f;
}
