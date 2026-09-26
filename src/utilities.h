#pragma once

#include "glm/glm.hpp"

#include <algorithm>
#include <istream>
#include <iterator>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#define PI 3.1415926535897932384626422832795028841971f
#define INV_PI 0.3183098861837906715377675267450287240689f
#define INV_FOUR_PI 0.07957747154594766788f
#define TWO_PI 6.2831853071795864769252867665590057683943f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define EPSILON 0.00001f

class GuiDataContainer {
  public:
    GuiDataContainer() : traced_depth(0) {}
    int traced_depth;
};

namespace utility_core {
extern void coordinate_system(glm::vec3 in_nor, glm::vec3 &out_tan,
                                          glm::vec3 &out_bit);
extern int divup(int x, int n);
extern float clamp(float f, float min, float max);
extern bool replace_string(std::string &str, const std::string &from,
                          const std::string &to);
extern glm::vec3 clamp_rgb(glm::vec3 color);
extern bool epsilon_check(float a, float b);
extern std::vector<std::string> tokenize_string(std::string str);
extern glm::mat4 build_transformation_matrix(glm::vec3 translation,
                                           glm::vec3 rotation, glm::vec3 scale);
extern std::string convert_int_to_string(int number);
extern std::istream &
safe_getline(std::istream &is,
            std::string &t); // Thanks to http://stackoverflow.com/a/6089413
} // namespace utility_core
