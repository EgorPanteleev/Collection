
#ifndef COLLECTION_COLOR_H
#define COLLECTION_COLOR_H
#undef RGB
class RGB {
public:
    void set( double _r, double _g, double _b );
    RGB();
    RGB( double _r, double _g, double _b);
    ~RGB();
public:
    double r;
    double g;
    double b;
};


#endif //COLLECTION_COLOR_H
