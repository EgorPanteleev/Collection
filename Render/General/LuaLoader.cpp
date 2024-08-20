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

float loadNumber( lua_State* L, const std::string& fieldName ) {
    float res;
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

float loadNumber( lua_State* L ) {
    float res;
    if (!lua_isnumber(L, -1)) {
        printf("Failed to load number.\n");
        lua_pop(L, 1);
        return __FLT_MAX__;
    }
    res = lua_tonumber(L, -1);
    lua_pop(L, 1);
    return res;
}

Vector<float> loadTable( lua_State* L, const std::string& fieldName ) {
    Vector<float> res;
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

Vector<float> loadTable( lua_State* L ) {
    Vector<float> res;
    for (int i = 0; i < lua_rawlen(L, -1); i++) {
        lua_rawgeti(L, -1, i + 1);
        res.push_back( loadNumber( L ) );
    }
    lua_pop(L, 1);
    return res;
}

Vector3f loadVector3f( lua_State* L, const std::string& fieldName ) {
    Vector<float> res = loadTable( L, fieldName );
    if ( res.size() < 3 ) return { __FLT_MAX__, __FLT_MAX__, __FLT_MAX__ };
    return { res[0], res[1], res[2] };
}

Vector3f loadVector3f( lua_State* L ) {
    Vector<float> res = loadTable( L );
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
    Vector3f color = loadVector3f( L );
    res.setColor( { color[0], color[1], color[2] } );
    Vector<float> props;
    for ( int i = 2; i <= lua_rawlen(L, -1); i++ ) {
        lua_rawgeti( L, -1 , i );
        props.push_back( loadNumber( L ) );
    }
    if ( props.size() == 2 ) {
        res.setDiffuse( props[0] );
        res.setReflection( props[1] );
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
    Vector3f moveTo = loadVector3f( L, "moveTo" );
    if ( moveTo != Vector3f( __FLT_MAX__, __FLT_MAX__, __FLT_MAX__ ) ) mesh->moveTo( moveTo );
    Vector3f move = loadVector3f( L, "move" );
    if ( move != Vector3f( __FLT_MAX__, __FLT_MAX__, __FLT_MAX__ ) ) mesh->move( move );
    Vector3f scaleTo = loadVector3f( L, "scaleTo" );
    if ( scaleTo != Vector3f( __FLT_MAX__, __FLT_MAX__, __FLT_MAX__ ) ) mesh->scaleTo( scaleTo );
    float scale = loadNumber( L, "scale" );
    if ( scale != __FLT_MAX__ ) mesh->scale( scale );
    Vector3f rotate = loadVector3f( L, "rotate" );
    if ( rotate != Vector3f( __FLT_MAX__, __FLT_MAX__, __FLT_MAX__ ) ) {
        mesh->rotate({ 1, 0, 0 }, rotate[0] );
        mesh->rotate({ 0, 1, 0 }, rotate[1] );
        mesh->rotate({ 0, 0, 1 }, rotate[2] );
    }
    Vector<float> minPoint = loadTable( L, "minPoint" );
    if ( minPoint.size() < 2 ) return;
    mesh->setMinPoint( { minPoint[0], minPoint[0], minPoint[0] }, (int) minPoint[1] );
    mesh->setMinPoint( { minPoint[0], minPoint[0], minPoint[0] }, (int) minPoint[1] );
}

void processMovement( lua_State* L, Sphere* sphere ) {
    Vector3f move = loadVector3f( L, "move" );
    sphere->move( move );
    Vector3f moveTo = loadVector3f( L, "moveTo" );
    sphere->moveTo( move );
    float scale = loadNumber( L, "scale" );
    sphere->scale( scale );
    Vector3f scaleTo = loadVector3f( L, "scaleTo" );
    sphere->scaleTo( scaleTo );
    Vector3f rotate = loadVector3f( L, "rotate" );
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
                Vector3f min = loadVector3f( L, "min" );
                Vector3f max = loadVector3f( L, "max" );
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
            Vector3f origin = loadVector3f(L, "origin");
            float radius = loadNumber( L, "radius" );
            Material material = loadMaterial( L );
            Sphere* sphere = new Sphere( radius, origin, material );
            //processMovement( L, &sphere );
            scene->add( *sphere );
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
    Vector3f pos = loadVector3f( L, "pos" );
    Vector3f lookAt = loadVector3f( L, "lookAt" );
    float FOV = loadNumber( L, "FOV" );
//    float aspectRatio = 1.6;
    float dV = w / 2 / tan( FOV * M_PI / 180 / 2 );
    return new Camera( pos, lookAt, dV, w, h );
}

Vector<float> loadSettings( lua_State* L ) {
    lua_getglobal(L, "RenderSettings");
    if (!lua_istable(L, -1)) {
        std::cerr << "Render settings not found." << std::endl;
        lua_pop(L, 1);
        return {};
    }
    float depth = loadNumber( L, "depth" );
    float ambientSamples = loadNumber( L, "ambientSamples" );
    float lightSamples = loadNumber( L, "lightSamples" );
    float denoise = loadNumber( L, "denoise" );
    Vector<float> res;
    res.push_back( depth );
    res.push_back( ambientSamples );
    res.push_back( lightSamples );
    res.push_back( denoise );
    return res;
}


