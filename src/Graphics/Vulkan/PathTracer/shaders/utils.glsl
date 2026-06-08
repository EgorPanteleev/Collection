#define FLT_MAX 3.402823e38
#define INT_MAX 2147483647
#define UINT_MAX 4294967295u
#define M_PI 3.14159265359
#define M_1_PI 0.3183098861837907

#define ALL_OBJECTS 0xFF
#define T_MIN 0.0
#define T_MAX 1e10

#define ORIGIN_TRESHOLD 0.03125
#define FLOAT_SCALE 1.0 / 65536.0
#define INT_SCALE 256.0

vec3 offsetRay(const vec3 p, const vec3 n) {
    ivec3 of_i = ivec3(INT_SCALE * n.x, INT_SCALE * n.y, INT_SCALE * n.z);
    vec3 p_i = vec3(intBitsToFloat(floatBitsToInt(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
    intBitsToFloat(floatBitsToInt(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
    intBitsToFloat(floatBitsToInt(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));
    return vec3(abs(p.x) < ORIGIN_TRESHOLD ? p.x + FLOAT_SCALE * n.x : p_i.x,
    abs(p.y) < ORIGIN_TRESHOLD ? p.y + FLOAT_SCALE * n.y : p_i.y,
    abs(p.z) < ORIGIN_TRESHOLD ? p.z + FLOAT_SCALE * n.z : p_i.z);
}

vec3 movedPoint(vec3 p, vec3 gn) {
    return offsetRay(p, gn);
}
