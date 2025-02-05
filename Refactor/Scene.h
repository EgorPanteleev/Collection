//
// Created by auser on 2/5/25.
//

#ifndef COLLECTION_SCENE_H
#define COLLECTION_SCENE_H
#include "BVH.h"
#include "Image.h"

class Scene: public BVH {
public:
    Scene(): background() {};

    void setSkyBox( const std::string& path ) { background.load( path ); }

    [[nodiscard]] HOST_DEVICE RGB getBackgroundColor( const Ray& ray, const RGB& defaultColor = { 0, 0, 0 } ) const {
        if ( background.empty() ) return defaultColor;


        float theta = acos(ray.direction[1]); // Polar angle (0 to pi)
        float phi = atan2(ray.direction[2], ray.direction[0]); // Azimuthal angle (-pi to pi)

        // Map spherical coordinates to UV coordinates
        float u = (phi + M_PI) / (2 * M_PI); // Normalize phi to [0, 1]
        float v = theta / M_PI; // Normalize theta to [0, 1]

        // Convert to pixel coordinates
        int x = static_cast<int>(u * background.width) % background.width;
        int y = static_cast<int>(v * background.height) % background.height;
        auto backColor = background.getPixelColor( x, y );
        double inv = 1.0 / 255;
        return { backColor[0] * inv, backColor[1] * inv, backColor[2] * inv };
    }

#if HIP_ENABLED
    HOST HittableList* copyToDevice() override;

    HOST HittableList* copyToHost() override;

    HOST void deallocateOnDevice() override;
#endif
public:
    Image background;
};


#endif //COLLECTION_SCENE_H
