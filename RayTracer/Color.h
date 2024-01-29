
#ifndef COLLECTION_COLOR_H
#define COLLECTION_COLOR_H
#undef RGB
class RGB {
public:
    void set( float _r, float _g, float _b );
    RGB operator+( const RGB& color ) const;
    RGB operator*( float a) const;
    RGB operator/( float a) const;
    RGB();
    RGB( float _r, float _g, float _b);
    ~RGB();
public:
    float r;
    float g;
    float b;
};


#endif //COLLECTION_COLOR_H
