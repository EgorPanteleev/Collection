//
// Created by auser on 7/29/24.
//

#ifndef COLLECTION_DENOISER_H
#define COLLECTION_DENOISER_H

#include "Canvas.h"

class Denoiser {
public:

    static void denoise( RGB** colorData, RGB** normalData, RGB** albedoData, int w, int h );

};


#endif //COLLECTION_DENOISER_H
