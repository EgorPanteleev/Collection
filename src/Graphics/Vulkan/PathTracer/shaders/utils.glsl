#define FLT_MAX 3.402823e38

struct BBox {
    vec4 min, max;
};

struct Ray {
    vec3 pos, dir, invDir;
    float tmin, tmax;
};

Ray makeRay(vec3 pos, vec3 dir, float tmin, float tmax) {
    Ray ray;
    ray.pos = pos;
    ray.dir = dir;
    ray.tmin = tmin;
    ray.tmax = tmax;
    ray.invDir = 1.0 / dir;
    return ray;
}

float intersect(BBox bbox, Ray ray) {
    vec3 tmin = (bbox.min.xyz - ray.pos) * ray.invDir;
    vec3 tmax = (bbox.max.xyz - ray.pos) * ray.invDir;

    vec3 t1 = min(tmin, tmax);
    vec3 t2 = max(tmin, tmax);

    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar  = min(min(t2.x, t2.y), t2.z);

    if (tFar >= max(tNear, ray.tmin) && tNear < ray.tmax)
    return tNear;
    return -1.0;
}
