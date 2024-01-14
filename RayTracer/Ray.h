#ifndef COLLECTION_RAY_H
#define COLLECTION_RAY_H
#include "Point.h"

class Ray {
public:
    Point getOrigin() const;
    Point getDirection() const;
    void setOrigin( Point orig );
    void setDirection( Point dir );
    Ray();
    Ray(Point from, Point to);
    ~Ray();
private:
    Point origin;
    Point direction;
};


#endif //COLLECTION_RAY_H
