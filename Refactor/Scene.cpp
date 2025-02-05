//
// Created by auser on 2/5/25.
//

#include "Scene.h"


#if HIP_ENABLED
HOST HittableList* Scene::copyToDevice() {
    auto device = HIP::allocateOnDevice<Scene>();

    device->hittables = move(*hittables.copyToDevice());
    device->indexes = move(*indexes.copyToDevice());
    device->bvhNodes = move(*bvhNodes.copyToDevice());
    device->background = *background.copyToDevice();
    return device;
}

HOST HittableList* Scene::copyToHost() {
    auto host = new Scene();
    HIP::copyToHost( host, this );

    host->hittables = move(*hittables.copyToHost());
    host->indexes = move(*indexes.copyToHost());
    host->bvhNodes = move(*bvhNodes.copyToHost());
    host->background = *background.copyToHost();
    return host;
}

HOST void Scene::deallocateOnDevice() {
    hittables.deallocateOnDevice();
    indexes.deallocateOnDevice();
    bvhNodes.deallocateOnDevice();
    background.deallocateOnDevice();

    HIP::deallocateOnDevice<Scene>( this );
}
#endif