//
// Created by auser on 12/31/24.
//

#include "LuaLoader.h"

#include "LuaLoader.h"


lua_State* Lua::newState() {
    auto luaState = luaL_newstate();
    luaL_openlibs(luaState);
    return luaState;
}

bool Lua::open( lua_State* luaState, const std::string& filePath ) {
    return luaL_dofile( luaState, filePath.c_str() ) == LUA_OK;
}

void Lua::close( lua_State* luaState ) {
    lua_close(luaState);
}

void Lua::getGlobal( lua_State* luaState, const std::string& name ) {
    lua_getglobal( luaState, name.c_str() );
}

void Lua::pop( lua_State* luaState, int idx ) {
    lua_pop( luaState, idx );
}

bool Lua::isNumber( lua_State* luaState, int idx ) {
    return lua_isnumber( luaState, idx );
}

bool Lua::isString( lua_State* luaState, int idx ) {
    return lua_isstring( luaState, idx );
}

bool Lua::isTable( lua_State* luaState, int idx ) {
    return lua_istable( luaState, idx );
}

void Lua::getField( lua_State* luaState, int idx, const std::string& name ) {
    lua_getfield( luaState, idx, name.c_str() );
}

void Lua::pushInteger( lua_State* luaState, int idx ) {
    lua_pushinteger( luaState, idx );
}

void Lua::pushNil( lua_State* luaState ) {
    lua_pushnil( luaState );
}

double Lua::toNumber( lua_State* luaState, int idx ) {
    return lua_tonumber( luaState, idx );
}

std::string Lua::toString( lua_State* luaState, int idx ) {
    return lua_tostring( luaState, idx );
}

double Lua::next( lua_State* luaState, int idx ) {
    return lua_next( luaState, idx );
}

size_t Lua::length( lua_State* luaState, int idx ) {
    return lua_rawlen(luaState, idx);
}

void Lua::getElement( lua_State* luaState, int idxTable, int idxElement ) {
    lua_rawgeti(luaState, idxTable, idxElement);
}

bool Lua::loadString( lua_State* luaState, std::string& str ) {
    if ( !isString( luaState, -1 ) ) {
        std::cout << "Failed to load string.\n";
        pop( luaState, 1);
        return false;
    }
    str = toString( luaState, -1 );
    pop( luaState, 1);
    return true;
}

bool Lua::loadString( lua_State* luaState, const std::string& name, std::string& str ) {
    getField( luaState, -1, name );
    if (!isString( luaState, -1 ) ) {
        std::cout << "Failed to load string.\n";
        pop( luaState, 1);
        return false;
    }
    str = toString( luaState, -1 );
    pop( luaState, 1);
    return true;
}

bool Lua::loadVec3d( lua_State* luaState, Vec3d& vec ) {
    if ( !isTable( luaState, -1 ) ) {
        std::cerr << "Vec3d must be table!\n";
        return false;
    }

    if ( length(luaState, -1) != 3 ) {
        std::cerr << "Vec3d must be a table with length 3!\n";
        return false;
    }

    for ( int i = 0; i < 3; ++i ) {
        getElement( luaState, -1, i + 1 );
        if ( !loadNumber( luaState, vec[i] ) ) return false;
    }

    pop( luaState, 1 );
    return true;
}

bool Lua::loadVec3d( lua_State* luaState, const std::string& name, Vec3d& vec ) {
    getField( luaState, -1, name );
    if ( !isTable( luaState, -1 ) ) {
        std::cerr << "Vec3d must be table!\n";
        return false;
    }

    if ( length(luaState, -1) != 3 ) {
        std::cerr << "Vec3d must be a table with length 3!\n";
        return false;
    }

    for ( int i = 0; i < 3; ++i ) {
        getElement( luaState, -1, i + 1 );
        if ( !loadNumber( luaState, vec[i] ) ) return false;
    }

    pop( luaState, 1 );
    return true;
}

bool Lua::loadRenderSettings( lua_State* luaState, Camera* cam ) {
    getGlobal( luaState, "Camera" );
    if ( !isTable( luaState, -1 ) ) {
        std::cerr << "Camera must be a table!\n";
        return false;
    }
    loadNumber( luaState, "aspectRatio", cam->aspectRatio );
    loadNumber( luaState, "imageWidth", cam->imageWidth );
    loadNumber( luaState, "samplesPerPixel", cam->samplesPerPixel );
    loadNumber( luaState, "maxDepth", cam->maxDepth );
    loadNumber( luaState, "vFov", cam->vFOV );
    loadNumber( luaState, "defocusAngle", cam->defocusAngle );
    loadNumber( luaState, "focusDistance", cam->focusDistance );
    loadVec3d( luaState, "lookFrom", cam->lookFrom );
    loadVec3d( luaState, "lookAt", cam->lookAt );
    loadVec3d( luaState, "globalUp", cam->globalUp );
    pop( luaState, 1 );
    return true;
}

bool Lua::loadMaterial( lua_State* luaState, Material*& material ) {
    getField( luaState, -1, "material" );
    if ( !isTable( luaState, -1 ) ) {
        std::cerr << "material must be a table!\n";
        return false;
    }
    getElement( luaState, -1, 1 );
    std::string type;

    if ( !loadString( luaState, type ) ) return false;

    if ( type == "LAMBERTIAN" ) {
        auto mat = new Lambertian();
        getElement( luaState, -1, 2 );
        if ( !loadVec3d( luaState, mat->albedo ) ) return false;
        material = mat;
        material->type = Material::LAMBERTIAN;
    } else if ( type == "METAL" ) {
        auto mat = new Metal();
        getElement( luaState, -1, 2 );
        if ( !loadVec3d( luaState, mat->albedo ) ) return false;
        getElement( luaState, -1, 3 );
        if ( !loadNumber( luaState, mat->fuzz ) ) return false;
        material = mat;
        material->type = Material::METAL;
    } else if ( type == "DIELECTRIC" ) {
        auto mat = new Dielectric();
        getElement( luaState, -1, 2 );
        if ( !loadNumber( luaState, mat->refractionIndex ) ) return false;
        material = mat;
        material->type = Material::DIELECTRIC;
    } else {
        std::cerr << "type " << type << " doesnt exist!\n";
        return false;
    }
    pop( luaState, 1 );
    return true;
}

bool Lua::loadSpheres( lua_State* luaState, HittableList* world ) {
    getField( luaState, -1, "Spheres" );
    if ( !isTable( luaState, -1 ) ) {
        std::cerr << "Spheres must be a table!\n";
        return false;
    }

    for ( int i = 0; i < length( luaState, -1 ); ++i ) {
        getElement( luaState, -1, i + 1 );
        if ( !isTable( luaState, -1 ) ) {
            std::cerr << "Spheres table elements must be a tables!\n";
            return false;
        }
        auto sphere = new Sphere();
        loadVec3d( luaState, "origin", sphere->origin );
        loadNumber( luaState, "radius", sphere->radius );
        loadMaterial( luaState, sphere->material );
        world->add( sphere );
        pop( luaState, 1 );
    }
    pop( luaState, 1 );
    return true;
}

bool Lua::loadWorld( lua_State* luaState, HittableList* world ) {
    getGlobal( luaState, "World" );
    if ( !isTable( luaState, -1 ) ) {
        std::cerr << "World must be a table!\n";
        return false;
    }
    if ( !loadSpheres( luaState, world ) ) return false;
    pop( luaState, 1 );
    return true;
}

//-----

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
//
//
//Canvas* loadCanvas( lua_State* L ) {
//    lua_getglobal(L, "Canvas");
//    if (!lua_istable(L, -1)) {
//        printf("No canvas given.\n");
//        return nullptr;
//    }
//    int w = (int) loadNumber( L, "w" );
//    int h = (int) loadNumber( L, "h" );
//    return new Canvas( w, h );
//}
//
//
//Material loadMaterial( lua_State* L ) {
//    Material res;
//    lua_getfield( L, -1, "material");
//    if (!lua_istable(L, -1)) {
//        std::cerr << "No material given." << std::endl;
//        lua_pop(L, 1);
//    }
//    lua_geti(L, -1, 1);
//    if (!lua_istable(L, -1)) {
//        std::cerr << "No color given." << std::endl;
//        lua_pop(L, 1);
//    }
//    Vec3d color = loadVec3d( L );
//    res.setColor( { color[0], color[1], color[2] } );
//    Vector<double> props;
//    for ( int i = 2; i <= lua_rawlen(L, -1); i++ ) {
//        lua_rawgeti( L, -1 , i );
//        props.push_back( loadNumber( L ) );
//    }
//    if ( props.size() == 2 ) {
//        res.setDiffuse( props[0] );
//        res.setRoughness( props[1] );
//    } else if ( props.size() == 1 ) {
//        res.setIntensity( props[0] );
//    }
//    lua_pop( L, 1 );
//    return res;
//}
//
//bool loadScene( lua_State* L, Scene* scene ) {
//    lua_getglobal(L, "Objects");
//    if (!lua_istable(L, -1)) {
//        std::cerr << "Objects not found." << std::endl;
//        lua_pop(L, 1);
//        return false;
//    }
//    loadSpheres( L, scene );
//    loadMeshes( L, scene );
//    //lua_pop(L, 1);
//    lua_getglobal(L, "Lights");
//    if (!lua_istable(L, -1)) {
//        std::cerr << "Lights not found." << std::endl;
//        lua_pop(L, 1);
//        return false;
//    }
//    loadSpheres( L, scene );
//    loadMeshes( L, scene );
//    return true;
//}
//
//void processMovement(lua_State* L, Mesh* mesh ) {
//    Vec3d moveTo = loadVec3d( L, "moveTo" );
//    if ( moveTo != Vec3d( __FLT_MAX__, __FLT_MAX__, __FLT_MAX__ ) ) mesh->moveTo( moveTo );
//    Vec3d move = loadVec3d( L, "move" );
//    if ( move != Vec3d( __FLT_MAX__, __FLT_MAX__, __FLT_MAX__ ) ) mesh->move( move );
//    Vec3d scaleTo = loadVec3d( L, "scaleTo" );
//    if ( scaleTo != Vec3d( __FLT_MAX__, __FLT_MAX__, __FLT_MAX__ ) ) mesh->scaleTo( scaleTo );
//    double scale = loadNumber( L, "scale" );
//    if ( scale != __FLT_MAX__ ) mesh->scale( scale );
//    Vec3d rotate = loadVec3d( L, "rotate" );
//    if ( rotate != Vec3d( __FLT_MAX__, __FLT_MAX__, __FLT_MAX__ ) ) {
//        mesh->rotate({ 1, 0, 0 }, rotate[0] );
//        mesh->rotate({ 0, 1, 0 }, rotate[1] );
//        mesh->rotate({ 0, 0, 1 }, rotate[2] );
//    }
//    Vector<double> minPoint = loadTable( L, "minPoint" );
//    if ( minPoint.size() < 2 ) return;
//    mesh->setMinPoint( { minPoint[0], minPoint[0], minPoint[0] }, (int) minPoint[1] );
//    mesh->setMinPoint( { minPoint[0], minPoint[0], minPoint[0] }, (int) minPoint[1] );
//}
//
//void processMovement( lua_State* L, Sphere* sphere ) {
//    Vec3d move = loadVec3d( L, "move" );
//    sphere->move( move );
//    Vec3d moveTo = loadVec3d( L, "moveTo" );
//    sphere->moveTo( move );
//    double scale = loadNumber( L, "scale" );
//    sphere->scale( scale );
//    Vec3d scaleTo = loadVec3d( L, "scaleTo" );
//    sphere->scaleTo( scaleTo );
//    Vec3d rotate = loadVec3d( L, "rotate" );
//    sphere->rotate({ 1, 0, 0 }, rotate[0] );
//    sphere->rotate({ 0, 1, 0 }, rotate[1] );
//    sphere->rotate({ 0, 0, 1 }, rotate[2] );
//}
//
//bool loadMeshes( lua_State* L, Scene* scene ) {
//    lua_getfield( L, -1, "Meshes");
//    if (!lua_istable(L, -1)) {
//        std::cerr << "No meshes given." << std::endl;
//        lua_pop(L, 1);
//        return false;
//    }
//    auto numMeshes = lua_rawlen(L, -1);
//    for (int i = 1; i <= numMeshes; i++) {
//        lua_rawgeti(L, -1, i);
//        if (lua_istable(L, -1)) {
//            std::string type = loadString(L, "Type");
//            Mesh* mesh = nullptr;
//            if ( type == "CubeMesh" ) {
//                Vec3d min = loadVec3d( L, "min" );
//                Vec3d max = loadVec3d( L, "max" );
//                Material material = loadMaterial( L );
//                mesh = new CubeMesh( min, max, material );
//            } else if ( type == "Mesh" ) {
//                std::string path = loadString(L, "Path");
//                auto* sks = new Mesh();
//                sks->loadMesh( path );
//                mesh = sks;
//            }
//            if ( !mesh ) continue;
//            processMovement( L, mesh );
//            scene->add( mesh );
//        }
//        lua_pop(L, 1);
//    }
//    lua_pop(L, 1);
//    return true;
//}
//
//bool loadSpheres( lua_State* L, Scene* scene ) {
//    lua_getfield( L, -1, "Spheres");
//    if (!lua_istable(L, -1)) {
//        std::cerr << "No spheres given." << std::endl;
//        lua_pop(L, 1);
//        return false;
//    }
//    auto numSpheres = lua_rawlen(L, -1);
//    for (int i = 1; i <= numSpheres; i++) {
//        lua_rawgeti(L, -1, i);
//        if (lua_istable(L, -1)) {
//            Vec3d origin = loadVec3d(L, "origin");
//            double radius = loadNumber( L, "radius" );
//            Material material = loadMaterial( L );
//            auto sphere = new Sphere( radius, origin, material );
//            //processMovement( L, &sphere );
//            scene->add( sphere );
//        }
//        lua_pop(L, 1);
//    }
//    lua_pop(L, 1);
//    return true;
//}
//
//Camera* loadCamera( lua_State* L, int w, int h ) {
//    lua_getglobal(L, "Camera");
//    if (!lua_istable(L, -1)) {
//        std::cerr << "Camera not found." << std::endl;
//        lua_pop(L, 1);
//        return nullptr;
//    }
//    Vec3d pos = loadVec3d( L, "pos" );
//    Vec3d lookAt = loadVec3d( L, "lookAt" );
//    double FOV = loadNumber( L, "FOV" );
////    double aspectRatio = 1.6;
//    double dV = w / 2 / tan( FOV * M_PI / 180 / 2 );
//    return new Camera( pos, lookAt, dV, w, h );
//}
//
//Vector<double> loadSettings( lua_State* L ) {
//    lua_getglobal(L, "RenderSettings");
//    if (!lua_istable(L, -1)) {
//        std::cerr << "Render settings not found." << std::endl;
//        lua_pop(L, 1);
//        return {};
//    }
//    double depth = loadNumber( L, "depth" );
//    double ambientSamples = loadNumber( L, "ambientSamples" );
//    double lightSamples = loadNumber( L, "lightSamples" );
//    double denoise = loadNumber( L, "denoise" );
//    Vector<double> res;
//    res.push_back( depth );
//    res.push_back( ambientSamples );
//    res.push_back( lightSamples );
//    res.push_back( denoise );
//    return res;
//}
//
//
