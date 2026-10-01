#include "glsl_utility.hpp"
#include "image.h"
#include "pathtrace.h"
#include "scene.h"
#include "camera.h"
#include "scene_structs.h"
#include "gui_data.h"
#include "utilities.h"
#include "math_utils.h"

#include <cstddef>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

static std::string start_time_string;

// For camera controls
static double last_x;
static double last_y;
static double last_time;
static constexpr float MAX_FRAME_DT = 0.12f;

static bool camchanged = true;

Camera camera;
Camera og_camera;
CameraController camera_controller;

Scene* scene;
GuiDataContainer* gui_data;
RenderState* render_state;
int iteration;

int width;
int height;

GLuint position_location = 0;
GLuint texcoords_location = 1;
GLuint display_image;
GLuint pbo;
cudaGraphicsResource_t cuda_pixel_resource;

GLFWwindow* window;
GuiDataContainer* imgui_data = NULL;
ImGuiIO* io = nullptr;
bool mouse_over_imgui_window = false;

// Forward declarations for window loop and interactivity
void run_cuda();
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
void mouse_position_callback(GLFWwindow* window, double xpos, double ypos);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

std::string current_time_string() {
    time_t now;
    time(&now);
    char buf[sizeof "0000-00-00_00-00-00z"];
    strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));
    return std::string(buf);
}

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

void init_textures() {
    glGenTextures(1, &display_image);
    glBindTexture(GL_TEXTURE_2D, display_image);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
}

void init_vao(void) {
    GLfloat vertices[] = {
        -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
    };

    GLfloat texcoords[] = {1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f};

    GLushort indices[] = {0, 1, 3, 3, 1, 2};

    GLuint vertex_buffer_obj_id[3];
    glGenBuffers(3, vertex_buffer_obj_id);

    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_obj_id[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)position_location, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(position_location);

    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_obj_id[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoords_location, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoords_location);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertex_buffer_obj_id[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint init_shader() {
    const char* attrib_locations[] = {"Position", "Texcoords"};
    GLuint program = glsl_utility::create_default_program(attrib_locations, 2);
    GLint location;

    // glUseProgram(program);
    if ((location = glGetUniformLocation(program, "u_image")) != -1) {
        glUniform1i(location, 0);
    }

    return program;
}

void delete_pbo(GLuint* pbo) {
    if (pbo) {
        if (cuda_pixel_resource) {
            cudaGraphicsUnregisterResource(cuda_pixel_resource);
        }
        glBindBuffer(GL_ARRAY_BUFFER, *pbo);
        glDeleteBuffers(1, pbo);

        *pbo = (GLuint)NULL;
    }
}

void delete_texture(GLuint* tex) {
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}

void cleanup_cuda() {
    if (pbo) {
        delete_pbo(&pbo);
    }
    if (display_image) {
        delete_texture(&display_image);
    }
}

void init_cuda() {
    cudaSetDevice(0);
}

void init_pbo() {
    // set up vertex data parameter
    int num_texels = width * height;
    int num_values = num_texels * 4;
    int texture_data_size = sizeof(GLubyte) * num_values;

    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1, &pbo);

    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, texture_data_size, NULL, GL_DYNAMIC_COPY);
    cudaGraphicsGLRegisterBuffer(&cuda_pixel_resource, pbo, cudaGraphicsRegisterFlagsWriteDiscard);
}

void error_callback(int error, const char* description) {
    fprintf(stderr, "%s\n", description);
}

bool init() {
    glfwSetErrorCallback(error_callback);

    if (!glfwInit()) {
        exit(EXIT_FAILURE);
    }

    window = glfwCreateWindow(width, height, "CIS 565 Path Tracer", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, mouse_position_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // Set up GL context
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        return false;
    }
    printf("Opengl Version:%s\n", glGetString(GL_VERSION));
    // Set up ImGui

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    io = &ImGui::GetIO();
    (void)io;
    ImGui::StyleColorsLight();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 120");

    // Initialize other stuff
    init_vao();
    init_textures();
    init_cuda();
    init_pbo();
    GLuint passthrough_program = init_shader();

    glUseProgram(passthrough_program);
    glActiveTexture(GL_TEXTURE0);

    return true;
}

void init_imgui_data(GuiDataContainer* gui_data) {
    imgui_data = gui_data;
}

// LOOK: Un-Comment to check ImGui Usage
void render_imgui() {
    mouse_over_imgui_window = io->WantCaptureMouse;

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    bool show_demo_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    static float f = 0.0f;
    static int counter = 0;

    ImGui::Begin(
        "Path Tracer Analytics"); // Create a window called "Hello, world!" and append into it.

    // LOOK: Un-Comment to check the output window and usage
    // ImGui::Text("This is some useful text.");               // Display some text (you can use a
    // format strings too) ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools
    // storing our window open/close state ImGui::Checkbox("Another Window", &show_another_window);

    // ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from
    // 0.0f to 1.0f ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats
    // representing a color

    // if (ImGui::Button("Button"))                            // Buttons return true when clicked
    // (most widgets return true when edited/activated)
    //     counter++;
    // ImGui::SameLine();
    // ImGui::Text("counter = %d", counter);
    ImGui::Text("Traced Depth %d", imgui_data->traced_depth);
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                ImGui::GetIO().Framerate);
    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

bool is_mouse_over_imgui_window() {
    return mouse_over_imgui_window;
}

void save_image() {
    copy_image_to_host();

    float samples = iteration;
    // output image file
    Image img(width, height);

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            int index = x + (y * width);
            glm::vec3 pix = render_state->image[index];
            img.set_pixel(width - 1 - x, y, glm::vec3(pix) / samples);
        }
    }

    std::string filename = render_state->image_name;
    std::ostringstream ss;
    ss << filename << "." << start_time_string << "." << samples << "samp";
    filename = ss.str();

    // CHECKITOUT
    img.save_png(filename);
    // img.save_hdr(filename);  // Save a Radiance HDR file
}

void main_loop() {
    pathtrace_init(scene);
    pathtrace_reset(scene);

    last_time = glfwGetTime();
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        double now = glfwGetTime();
        float dt = glm::min((float)(now - last_time), MAX_FRAME_DT);
        last_time = now;
        camchanged |= camera_controller.update_camera(camera, dt);

        run_cuda();

        std::string title =
            "CIS565 Path Tracer | " + utility_core::convert_int_to_string(iteration) + " Iterations";
        glfwSetWindowTitle(window, title.c_str());
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, display_image);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glClear(GL_COLOR_BUFFER_BIT);

        // Binding GL_PIXEL_UNPACK_BUFFER back to default
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // VAO, shader program, and texture already bound
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);

        // Render ImGui Stuff
        render_imgui();

        glfwSwapBuffers(window);
    }

    save_image();

    pathtrace_free();
    cleanup_cuda();
    cudaDeviceReset();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv) {
    start_time_string = current_time_string();

    if (argc < 2) {
        printf("Usage: %s SCENEFILE.json\n", argv[0]);
        return 1;
    }

    const char* scene_file = argv[1];

    // Load scene file
    scene = new Scene(scene_file);

    // Create ImGui data instance
    gui_data = new GuiDataContainer();

    // Set up camera stuff from loaded path tracer settings
    iteration = 0;
    render_state = &scene->state;
    const CameraData& cam = render_state->camera;
    width = cam.resolution.x;
    height = cam.resolution.y;

    camera.position = cam.position;
    camera.yaw = glm::atan(-cam.view.x, -cam.view.z);
    camera.pitch = glm::asin(cam.view.y);
    og_camera = camera;

    // Initialize CUDA and GL components
    init();

    // Initialize ImGui Data
    init_imgui_data(gui_data);
    init_data_container(gui_data);

    // GLFW main loop
    main_loop();

    return 0;
}

void reset_accumulation() {
    iteration = 0;
    camera.write_data(render_state->camera);
    camchanged = false;
}

void run_cuda() {
    if (camchanged) {
        reset_accumulation();
        pathtrace_reset(scene);
    }

    if (iteration < render_state->iterations) {
        cudaGraphicsMapResources(1, &cuda_pixel_resource);

        uchar4* pbo_dptr;
        size_t bytes;
        iteration++;

        cudaGraphicsResourceGetMappedPointer((void**)&pbo_dptr, &bytes, cuda_pixel_resource);

        // execute the kernel
        pathtrace(pbo_dptr, iteration);

        // unmap buffer object
        cudaGraphicsUnmapResources(1, &cuda_pixel_resource);
    } else {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
}

//-------------------------------
//------INTERACTIVITY SETUP------
//-------------------------------

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (camera_controller.process_keyboard(key, action != GLFW_RELEASE)) {
        return;
    }

    if (action == GLFW_PRESS) {
        switch (key) {
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, GL_TRUE);
            break;
        case GLFW_KEY_S:
            save_image();
            break;
        case GLFW_KEY_SPACE:
            camera = og_camera;
            camchanged = true;
            break;
        }
    }
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button != GLFW_MOUSE_BUTTON_RIGHT || (action == GLFW_PRESS && is_mouse_over_imgui_window())) {
        return;
    }

    bool captured = action == GLFW_PRESS;
    camera_controller.set_captured(captured);
    glfwSetInputMode(window, GLFW_CURSOR, captured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
    glfwGetCursorPos(window, &last_x, &last_y);
}

void mouse_position_callback(GLFWwindow* window, double xpos, double ypos) {
    camera_controller.handle_mouse(xpos - last_x, ypos - last_y);
    last_x = xpos;
    last_y = ypos;
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    if (!is_mouse_over_imgui_window()) {
        camera_controller.handle_mouse_scroll(yoffset);
    }
}
