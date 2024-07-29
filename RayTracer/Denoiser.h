//
// Created by auser on 7/29/24.
//

#ifndef COLLECTION_DENOISER_H
#define COLLECTION_DENOISER_H

#include "Canvas.h"

class Denoiser {
public:

    Denoiser( Canvas* canvas );

    void denoise();

private:

    Canvas* canvas;

};


#endif //COLLECTION_DENOISER_H
