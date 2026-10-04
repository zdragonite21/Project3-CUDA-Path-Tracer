#pragma once

#include "camera.h"
#include "render_settings.h"
#include "scene_structs.h"
#include <string>
#include <vector>

struct RenderState {
    CameraData camera;
    CameraConfig controls;
    unsigned int iterations;
    std::vector<glm::vec3> image;
    std::string image_name;
    RenderSettings settings;
};

struct EnvironmentMap {
    int width = 0;
    int height = 0;
    std::string path;
    std::vector<glm::vec3> pixels;
};

class Scene {
  private:
    void load_hdri_pixels(const std::string& json_name, const std::string& hdri_path);
    void load_from_json(const std::string& json_name);

  public:
    Scene(std::string filename);
    void save_to_json(const std::string& out_path) const;

    std::string filename;

    std::vector<Geom> geoms;
    std::vector<std::string> material_names;
    std::vector<Material> materials;
    std::vector<Light> lights;
    RenderState state;
    EnvironmentMap env;
};
