//
// Created by auser on 8/5/24.
//

#include "LuaLoader.h"

std::string loadString( lua_State* L, const std::string& fieldName ) {
    std::string res;
    lua_getfield(L, -1, fieldName.c_str() );
    if (!lua_isstring(L, -1)) {
        printf("Failed to load string in field %s.\n", fieldName.c_str() );
        lua_pop(L, 1);
        return "";
    }
    res = lua_tostring(L, -1);
    lua_pop(L, 1);
    return res;
}

double loadNumber( lua_State* L, const std::string& fieldName ) {
    double res;
    lua_getfield(L, -1, fieldName.c_str() );
    if (!lua_isnumber(L, -1)) {
        printf("Failed to load number in field %s.\n", fieldName.c_str() );
        lua_pop(L, 1);
        return __FLT_MAX__;
    }
    res = lua_tonumber(L, -1);
    lua_pop(L, 1);
    return res;
}

double loadNumber( lua_State* L ) {
    double res;
    if (!lua_isnumber(L, -1)) {
        printf("Failed to load number.\n");
        lua_pop(L, 1);
        return __FLT_MAX__;
    }
    res = lua_tonumber(L, -1);
    lua_pop(L, 1);
    return res;
}

Vector<double> loadTable( lua_State* L, const std::string& fieldName ) {
    Vector<double> res;
    lua_getfield(L, -1, fieldName.c_str() );
    if (!lua_istable(L, -1)) {
        printf("Failed to load table in field %s.\n", fieldName.c_str() );
        lua_pop(L, 1);
        return {};
    }
    for (int i = 0; i < lua_rawlen(L, -1); i++) {
        lua_rawgeti(L, -1, i + 1);
        res.push_back( loadNumber( L ) );
    }
    lua_pop(L, 1);
    return res;
}

Vector<double> loadTable( lua_State* L ) {
    Vector<double> res;
    for (int i = 0; i < lua_rawlen(L, -1); i++) {
        lua_rawgeti(L, -1, i + 1);
        res.push_back( loadNumber( L ) );
    }
    lua_pop(L, 1);
    return res;
}

Vec3d loadVec3d( lua_State* L, const std::string& fieldName ) {
    Vector<double> res = loadTable( L, fieldName );
    if ( res.size() < 3 ) return { __FLT_MAX__, __FLT_MAX__, __FLT_MAX__ };
    return { res[0], res[1], res[2] };
}

Vec3d loadVec3d( lua_State* L ) {
    Vector<double> res = loadTable( L );
    if ( res.size() < 3 ) return {};
    return { res[0], res[1], res[2] };
}


Canvas* loadCanvas( lua_State* L ) {
    lua_getglobal(L, "Canvas");
    if (!lua_istable(L, -1)) {
        printf("No canvas given.\n");
        return nullptr;
    }
    int w = (int) loadNumber( L, "w" );
    int h = (int) loadNumber( L, "h" );
    return new Canvas( w, h );
}


Material loadMaterial( lua_State* L ) {
    Material res;
    lua_getfield( L, -1, "material");
    if (!lua_istable(L, -1)) {
        std::cerr << "No material given." << std::endl;
        lua_pop(L, 1);
    }
    lua_geti(L, -1, 1);
    if (!lua_istable(L, -1)) {
        std::cerr << "No color given." << std::endl;
        lua_pop(L, 1);
    }
    Vec3d color = loadVec3d( L );
    res.setColor( { color[0], color[1], color[2] } );
    Vector<double> props;
    for ( int i = 2; i <= lua_rawlen(L, -1); i++ ) {
        lua_rawgeti( L, -1 , i );
        props.push_back( loadNumber( L ) );
    }
    if ( props.size() == 2 ) {
        res.setDiffuse( props[0] );
        res.setRoughness( props[1] );
    } else if ( props.size() == 1 ) {
        res.setIntensity( props[0] );
    }
    lua_pop( L, 1 );
    return res;
}

bool loadScene( lua_State* L, Scene* scene ) {
    lua_getglobal(L, "Objects");
    if (!lua_istable(L, -1)) {
        std::cerr << "Objects not found." << std::endl;
        lua_pop(L, 1);
        return false;
    }
    loadSpheres( L, scene );
    loadMeshes( L, scene );
    //lua_pop(L, 1);
    lua_getglobal(L, "Lights");
    if (!lua_istable(L, -1)) {
        std::cerr << "Lights not found." << std::endl;
        lua_pop(L, 1);
        return false;
    }
    loadSpheres( L, scene );
    loadMeshes( L, scene );
    return true;
}

void processMovement(lua_State* L, Mesh* mesh ) {
    Vec3d moveTo = loadVec3d( L, "moveTo" );
    if ( moveTo != Vec3d( __FLT_MAX__, __FLT_MAX__, __FLT_MAX__ ) ) mesh->moveTo( moveTo );
    Vec3d move = loadVec3d( L, "move" );
    if ( move != Vec3d( __FLT_MAX__, __FLT_MAX__, __FLT_MAX__ ) ) mesh->move( move );
    Vec3d scaleTo = loadVec3d( L, "scaleTo" );
    if ( scaleTo != Vec3d( __FLT_MAX__, __FLT_MAX__, __FLT_MAX__ ) ) mesh->scaleTo( scaleTo );
    double scale = loadNumber( L, "scale" );
    if ( scale != __FLT_MAX__ ) mesh->scale( scale );
    Vec3d rotate = loadVec3d( L, "rotate" );
    if ( rotate != Vec3d( __FLT_MAX__, __FLT_MAX__, __FLT_MAX__ ) ) {
        mesh->rotate({ 1, 0, 0 }, rotate[0] );
        mesh->rotate({ 0, 1, 0 }, rotate[1] );
        mesh->rotate({ 0, 0, 1 }, rotate[2] );
    }
    Vector<double> minPoint = loadTable( L, "minPoint" );
    if ( minPoint.size() < 2 ) return;
    mesh->setMinPoint( { minPoint[0], minPoint[0], minPoint[0] }, (int) minPoint[1] );
    mesh->setMinPoint( { minPoint[0], minPoint[0], minPoint[0] }, (int) minPoint[1] );
}

void processMovement( lua_State* L, Sphere* sphere ) {
    Vec3d move = loadVec3d( L, "move" );
    sphere->move( move );
    Vec3d moveTo = loadVec3d( L, "moveTo" );
    sphere->moveTo( move );
    double scale = loadNumber( L, "scale" );
    sphere->scale( scale );
    Vec3d scaleTo = loadVec3d( L, "scaleTo" );
    sphere->scaleTo( scaleTo );
    Vec3d rotate = loadVec3d( L, "rotate" );
    sphere->rotate({ 1, 0, 0 }, rotate[0] );
    sphere->rotate({ 0, 1, 0 }, rotate[1] );
    sphere->rotate({ 0, 0, 1 }, rotate[2] );
}

bool loadMeshes( lua_State* L, Scene* scene ) {
    lua_getfield( L, -1, "Meshes");
    if (!lua_istable(L, -1)) {
        std::cerr << "No meshes given." << std::endl;
        lua_pop(L, 1);
        return false;
    }
    auto numMeshes = lua_rawlen(L, -1);
    for (int i = 1; i <= numMeshes; i++) {
        lua_rawgeti(L, -1, i);
        if (lua_istable(L, -1)) {
            std::string type = loadString(L, "Type");
            Mesh* mesh = nullptr;
            if ( type == "CubeMesh" ) {
                Vec3d min = loadVec3d( L, "min" );
                Vec3d max = loadVec3d( L, "max" );
                Material material = loadMaterial( L );
                mesh = new CubeMesh( min, max, material );
            } else if ( type == "Mesh" ) {
                std::string path = loadString(L, "Path");
                auto* sks = new Mesh();
                sks->loadMesh( path );
                mesh = sks;
            }
            if ( !mesh ) continue;
            processMovement( L, mesh );
            scene->add( mesh );
        }
        lua_pop(L, 1);
    }
    lua_pop(L, 1);
    return true;
}

bool loadSpheres( lua_State* L, Scene* scene ) {
    lua_getfield( L, -1, "Spheres");
    if (!lua_istable(L, -1)) {
        std::cerr << "No spheres given." << std::endl;
        lua_pop(L, 1);
        return false;
    }
    auto numSpheres = lua_rawlen(L, -1);
    for (int i = 1; i <= numSpheres; i++) {
        lua_rawgeti(L, -1, i);
        if (lua_istable(L, -1)) {
            Vec3d origin = loadVec3d(L, "origin");
            double radius = loadNumber( L, "radius" );
            Material material = loadMaterial( L );
            auto sphere = new Sphere( radius, origin, material );
            //processMovement( L, &sphere );
            scene->add( sphere );
        }
        lua_pop(L, 1);
    }
    lua_pop(L, 1);
    return true;
}

Camera* loadCamera( lua_State* L, int w, int h ) {
    lua_getglobal(L, "Camera");
    if (!lua_istable(L, -1)) {
        std::cerr << "Camera not found." << std::endl;
        lua_pop(L, 1);
        return nullptr;
    }
    Vec3d pos = loadVec3d( L, "pos" );
    Vec3d lookAt = loadVec3d( L, "lookAt" );
    double FOV = loadNumber( L, "FOV" );
//    double aspectRatio = 1.6;
    double dV = w / 2 / tan( FOV * M_PI / 180 / 2 );
    return new Camera( pos, lookAt, dV, w, h );
}

Vector<double> loadSettings( lua_State* L ) {
    lua_getglobal(L, "RenderSettings");
    if (!lua_istable(L, -1)) {
        std::cerr << "Render settings not found." << std::endl;
        lua_pop(L, 1);
        return {};
    }
    double depth = loadNumber( L, "depth" );
    double ambientSamples = loadNumber( L, "ambientSamples" );
    double lightSamples = loadNumber( L, "lightSamples" );
    double denoise = loadNumber( L, "denoise" );
    Vector<double> res;
    res.push_back( depth );
    res.push_back( ambientSamples );
    res.push_back( lightSamples );
    res.push_back( denoise );
    return res;
}


