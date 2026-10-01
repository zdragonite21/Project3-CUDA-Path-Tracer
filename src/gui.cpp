#include "gui.h"
#include "gui_data.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "render_settings.h"
#include "scene.h"
#include <cmath>
#include <cstdio>

static bool slider_log10(const char* label, float* v, float exp_min, float exp_max) {
    float e = std::log10(*v);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.2e", *v);
    if (!ImGui::SliderFloat(label, &e, exp_min, exp_max, buf)) return false;
    *v = std::pow(10.f, e);
    return true;
}

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

bool gui::render_imgui(GuiRefs& refs) {
    GuiDataContainer& d = *refs.data;
    RenderSettings& s = refs.scene->state.settings;
    bool reset = false;

    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);

    ImGui::Begin(
        "Path Tracer Analytics"); // Create a window called "Hello, world!" and append into it.

    reset |= ImGui::SliderInt("max depth", &s.max_depth, 1, 64);
    reset |= ImGui::SliderInt("max steps", &s.sdf_max_steps, 1, 512);
    reset |= slider_log10("sdf hit eps", &s.sdf_hit_eps, -6.f, -1.f);
    reset |= slider_log10("sdf normal eps", &s.sdf_normal_eps, -6.f, -1.f);
    reset |= ImGui::SliderFloat("env strength", &s.env_strength, 0.f, 10.f);
    ImGui::Checkbox("agx tonemapping", &s.agx);

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
