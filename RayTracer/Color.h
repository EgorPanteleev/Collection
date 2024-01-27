
#ifndef COLLECTION_COLOR_H
#define COLLECTION_COLOR_H
#undef RGB
class RGB {
public:
    void set( float _r, float _g, float _b );
    RGB operator*( double a) const;
    RGB operator/( double a) const;
    RGB();
    RGB( double _r, double _g, double _b);
    ~RGB();
public:
    double r;
    double g;
    double b;
};


#endif //COLLECTION_COLOR_H
