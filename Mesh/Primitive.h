////
//// Created by auser on 8/20/24.
////
//
//#ifndef COLLECTION_PRIMITIVE_H
//#define COLLECTION_PRIMITIVE_H
//#include "Triangle.h"
//#include "Sphere.h"
//
//class Primitive {
//public:
//    Primitive();
//    Primitive( Triangle* tri );
//    Primitive( Sphere* sph );
//    enum Type {
//        TRIANGLE,
//        SPHERE,
//        UNKNOWN
//    };
//    Type getType() const;
//    bool operator==( const Primitive& other );
//    bool operator!=( const Primitive& other );
//    void rotate( const Vector3f& axis, float angle );
//    void move( const Vector3f& p );
//    void moveTo( const Vector3f& point );
//    void scale( float scaleValue );
//    void scale( const Vector3f& scaleVec );
//    void scaleTo( float scaleValue );
//    void scaleTo( const Vector3f& scaleVec );
//    void setMaterial( const Material& mat );
//    [[nodiscard]] Vector3f getSamplePoint() const;
//    [[nodiscard]] BBox getBBox() const;
//    [[nodiscard]] Vector3f getOrigin() const;
//    [[nodiscard]] bool isContainPoint( const Vector3f& p ) const;
//    [[nodiscard]] float intersectsWithRay( const Ray& ray ) const;
//    [[nodiscard]] Vector3f getNormal( const Vector3f& P ) const;
//    [[nodiscard]] RGB getColor( const Vector3f& P ) const;
//    [[nodiscard]] RGB getAmbient( const Vector3f& P ) const;
//    [[nodiscard]] float getRoughness( const Vector3f& P ) const;
//    [[nodiscard]] Material getMaterial() const;
//private:
//    Type type;
//    Triangle* triangle;
//    Sphere* sphere;
//};
//
//
//#endif //COLLECTION_PRIMITIVE_H
