#include "gui.h"
#include "gui_data.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "render_settings.h"
#include "scene.h"
#include "bsdf/bsdf_gui.cuh"
#include <cfloat>

void gui::init(GLFWwindow* window) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 120");
}

void gui::begin_frame(bool& mouse_over_imgui_window) {
    mouse_over_imgui_window = ImGui::GetIO().WantCaptureMouse;

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

static bool render_section(RenderSettings& s) {
    bool reset = false;
    reset |= ImGui::SliderInt("max depth", &s.max_depth, 1, 64);
    reset |= ImGui::SliderFloat("env strength", &s.env_strength, 0.f, 10.f);
    return reset;
}
static bool camera_section(CameraData& cam) {
    bool reset = false;
    reset |= ImGui::SliderFloat("lens radius", &cam.lens_radius, 0.f, 5.f);
    reset |= ImGui::DragFloat("focal distance", &cam.focal_distance, 0.5f, 0.f, FLT_MAX);
    float fovy = cam.fov.y;
    reset |= ImGui::SliderFloat("fov", &fovy, 10.f, 170.f);
    cam.set_fov(fovy);
    return reset;
}

static bool sdf_section(RenderSettings& s) {
    bool reset = false;
    reset |= ImGui::SliderInt("max steps", &s.sdf_max_steps, 1, 512);
    reset |= ImGui::SliderFloat("sdf hit eps", &s.sdf_hit_eps, 1e-6f, 1e-1f, "%.6f",
                                ImGuiSliderFlags_Logarithmic);
    reset |= ImGui::SliderFloat("sdf normal eps", &s.sdf_normal_eps, 1e-6f, 1e-1f, "%.6f",
                                ImGuiSliderFlags_Logarithmic);
    return reset;
}

static bool materials_section(Scene& scene) {
    bool reset = false;
    for (int i = 0; i < scene.materials.size(); ++i) {
        Material& mat = scene.materials[i];
        ImGui::PushID(i);
        if (ImGui::TreeNode(scene.material_names[i].c_str())) {
            reset |= cuda::std::visit([](auto& b) { return draw_bsdf(b); }, mat.bsdf);
            reset |= ImGui::ColorEdit3("emission", &mat.emission.x,
                                         ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float);
            ImGui::TreePop();
        }
        ImGui::PopID();
    }
    return reset;
}

static void display_section(RenderSettings& s) {
    ImGui::Checkbox("agx tonemapping", &s.agx);
}
static void controls_section(CameraConfig& c) {
    ImGui::SliderFloat("aceeleration", &c.accel, 1.f, 10000.f, "%4f", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("damping", &c.damping, 0.f, 200.f);
    ImGui::SliderFloat("mouse sens", &c.mouse_sens, 0.01f, 3.f);
    ImGui::SliderFloat("scroll sens", &c.scroll_sens, 0.001f, 3.f);
}

bool gui::render_imgui(GuiRefs& refs) {
    GuiDataContainer& d = *refs.data;
    RenderSettings& s = refs.scene->state.settings;
    bool reset = false;

    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);

    ImGui::Begin(
        "Path Tracer Analytics"); // Create a window called "Hello, world!" and append into it.

    if (ImGui::CollapsingHeader("Render", ImGuiTreeNodeFlags_DefaultOpen)) {
        reset |= render_section(s);
    }
    if (ImGui::CollapsingHeader("SDF", ImGuiTreeNodeFlags_DefaultOpen)) {
        reset |= sdf_section(s);
    }
    if (ImGui::CollapsingHeader("Materials")) {
        reset |= materials_section(*refs.scene);
    }
    if (ImGui::CollapsingHeader("Camera")) {
        reset |= camera_section(refs.scene->state.camera);
    }
    if (ImGui::CollapsingHeader("Display")) {
        display_section(s);
    }
    if (ImGui::CollapsingHeader("Controls")) {
        controls_section(refs.camera->settings);
    }

    // LOOK: Un-Comment to check the output window and usage
    // ImGui::Text("This is some useful text.");               // Display some text (you can use a
    // format strings too) ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools
    // storing our window open/close state ImGui::Checkbox("Another Window", &show_another_window);

    // ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from
    // 0.0f to 1.0f ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats
    // representing a color

    // if (ImGui::Button("Button"))                            // Buttons return true when clicked
    // // (most widgets return true when edited/activated)
    //     counter++;
    // ImGui::SameLine();
    // ImGui::Text("counter = %d", counter);
    ImGui::Text("Traced Depth %d", refs.data->traced_depth);
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                ImGui::GetIO().Framerate);
    ImGui::End();

    return reset;
}

void gui::end_frame() {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void gui::shutdown() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}
