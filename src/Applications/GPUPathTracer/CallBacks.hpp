//
// Created by igor on 4/18/26.
//

#ifndef COLLECTION_CALLBACKS_HPP
#define COLLECTION_CALLBACKS_HPP

#include "Window.hpp"
#include "Camera.hpp"

namespace cvk = crv::graphics::vulkan;
namespace cs = crv::scene;

void setCallBacks(cvk::Window& window, cs::AbsCamera* camera);


#endif //COLLECTION_CALLBACKS_HPP