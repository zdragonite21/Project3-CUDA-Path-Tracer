#pragma once

#include "camera.h"
#include "scene_structs.h"
#include <vector>
#include <string>

struct RenderState {
    CameraData camera;
    unsigned int iterations;
    int trace_depth;
    std::vector<glm::vec3> image;
    std::string image_name;
};

struct EnvironmentMap {
    int width = 0;
    int height = 0;
    float strength = 1.f;
    int light_idx = -1;
    std::vector<glm::vec3> pixels;
};

class Scene
{
  private:
    void load_hdri_pixels(const std::string& json_name, const std::string& hdri_path);
    void load_from_json(const std::string& json_name);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Light> lights;
    RenderState state;
    EnvironmentMap env;
};
