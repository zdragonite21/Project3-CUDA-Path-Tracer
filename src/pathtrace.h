#pragma once

#include "scene.h"
#include "utilities.h"

void init_data_container(GuiDataContainer* guiData);
void pathtrace_init(Scene* scene);
void pathtrace_reset(Scene* scene);
void pathtrace_free();
void pathtrace(uchar4 *pbo, int iteration);
void copy_image_to_host();
