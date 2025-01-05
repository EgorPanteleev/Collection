
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include "hip/hip_runtime.h"
#include <hip/hip_runtime_api.h>
#include <chrono>
#include "Vector.h"
#include "Sphere.h"
#include "Triangle.h"
#include "Camera.h"
#include "Vec3.h"
#include "RGB.h"
#include "Vec2.h"
#include "Kernel.h"
#include "LuaLoader.h"

//#include <hip/hip_gl_interop.h>

//TODO LUA!

const int WIDTH = 800, HEIGHT = 500;
unsigned char* buffer = new unsigned char[WIDTH * HEIGHT * 3];  // RGB buffer

dim3 blockSize(16, 16);
dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (HEIGHT + blockSize.y - 1) / blockSize.y);
bool firstMouse = true;
Point2d prevPos;
Point2d offset;
bool pressed = false;
bool released = false;
double* memory = nullptr;
int numFrames = 0;
bool clear;
double oldDefocusAngle = 0;
bool keyStateF = false;

void clearAll() {
    numFrames = 0;
    clearMemory<<<gridSize, blockSize>>>( WIDTH, HEIGHT, memory );
}

void addCube( HittableList* world, Material* mat, const Vec3d& v1, const Vec3d& v2, Material* left = nullptr, Material* right = nullptr ) {
    Vec3d vertices[8];
    vertices[0] = { v1[0], v1[1], v1[2] };
    vertices[1] = { v1[0], v1[1], v2[2] };
    vertices[2] = { v1[0], v2[1], v1[2] };
    vertices[3] = { v1[0], v2[1], v2[2] };
    vertices[4] = { v2[0], v1[1], v1[2] };
    vertices[5] = { v2[0], v1[1], v2[2] };
    vertices[6] = { v2[0], v2[1], v1[2] };
    vertices[7] = { v2[0], v2[1], v2[2] };

    if ( !left ) left = mat;
    if ( !right ) right = mat;

    // Add triangles for each face of the cube
    world->add(new Triangle(vertices[0], vertices[1], vertices[3], left)); // left face
    world->add(new Triangle(vertices[0], vertices[3], vertices[2], left));

    world->add(new Triangle(vertices[4], vertices[6], vertices[7], right)); // Right face
    world->add(new Triangle(vertices[4], vertices[7], vertices[5], right));

    world->add(new Triangle(vertices[0], vertices[2], vertices[6], mat)); // Back face
    world->add(new Triangle(vertices[0], vertices[6], vertices[4], mat));

//    world->add(new Triangle(vertices[1], vertices[5], vertices[7], mat)); // Front face
//    world->add(new Triangle(vertices[1], vertices[7], vertices[3], mat));

    world->add(new Triangle(vertices[0], vertices[4], vertices[5], mat)); // Bottom face
    world->add(new Triangle(vertices[0], vertices[5], vertices[1], mat));

    world->add(new Triangle(vertices[2], vertices[3], vertices[7], mat)); // Top face
    world->add(new Triangle(vertices[2], vertices[7], vertices[6], mat));
}

void quads( HittableList* world ) {
    //Materials
    Material* ground = new Lambertian({ 0.8, 0.8, 0.8 } );
    Material* red = new Lambertian({ 0.8, 0.2, 0.2 } );
    Material* yellow = new Light({ 0.8, 0.8, 0.2 }, 2 );
    Material* blue = new Metal({ 0.2, 0.2, 0.8 }, 0.1 );
    //Hittables
    world->add( new Sphere( 1000, { 0, -1000, 0 }, ground ));
    //world->add( new Triangle( {-2, 1, -2 }, { -1.5, 2, -2 }, { 1, 1, -2 }, red ));
    addCube( world, red, { -0.5, 0.5, -1  }, { 0.5, 1.5, -2 } );
    addCube( world, yellow, { -2.5, 0.5, -1  }, { -1.5, 1.5, -2 } );
    addCube( world, blue, { 1.5, 0.5, -1  }, { 2.5, 1.5, -2 } );
}

void room( HittableList* world ) {
    //Materials
    Material* ground = new Lambertian({ 0.8, 0.8, 0.8 } );
    Material* red = new Lambertian({ 0.8, 0.2, 0.2 } );
    Material* green = new Lambertian({ 0.2, 0.8, 0.2 } );
    Material* gray = new Lambertian({ 0.8, 0.8, 0.8 } );
    Material* light = new Light({ 1, 1, 1 }, 2 );
    //Hittables
    world->add( new Sphere( 1000, { 0, -1000, 0 }, ground ));
    //world->add( new Triangle( {-2, 1, -2 }, { -1.5, 2, -2 }, { 1, 1, -2 }, red ));
    double length = 5;
    double width = 5;
    double height = 5;
    addCube( world, gray, { -width/2, 0, -length / 2  }, { width/2, height, length / 2 }, green, red );
    length = 2;
    width = 2;
    height = 5;
    addCube( world, light, { -width/2, height - 0.1, -length / 2  }, { width/2, height - 0.01, length / 2 } );
//    addCube( world, yellow, { -2.5, 0.5, -1  }, { -1.5, 1.5, -2 } );
//    addCube( world, blue, { 1.5, 0.5, -1  }, { 2.5, 1.5, -2 } );
}


BVH* initDeviceWorld( lua_State* luaState ) {
    BVH world;

    //Lua::loadWorld( luaState, &world );

//    quads( &world );
    room( &world );

    world.build();

    auto worldDevice = (BVH*) world.copyToDevice();
    return worldDevice;
}

Camera* initDeviceCamera( lua_State* luaState ) {
    Camera cam;

    Lua::loadRenderSettings( luaState, &cam );

    cam.init();

    oldDefocusAngle = cam.defocusAngle;

    auto deviceCamera = HIP::allocateOnDevice<Camera>();

    HIP::copyToDevice( &cam, deviceCamera );
    return deviceCamera;
}

void finalize( BVH* world, Camera* cam, unsigned char* deviceBuffer ) {
    world->deallocateOnDevice();
    HIP::deallocateOnDevice( cam );
    HIP::deallocateOnDevice( deviceBuffer );
    HIP::deallocateOnDevice( memory );
}

hiprandState* initStates( int width, int height ) {
    hiprandState *states;
    HIP_ASSERT(hipMalloc((void **)&states, width * height * sizeof(hiprandState)));
    initStates<<<gridSize, blockSize>>>(width, height, 1984, states);
    return states;
}


void updateBuffer( Camera* cam, BVH* world, unsigned char* deviceBuffer, hiprandState* states ) {
    auto start =  std::chrono::steady_clock::now();
    render<<<gridSize, blockSize>>>( cam, world, deviceBuffer, memory, 1.0 / numFrames, states );
    initStates<<<gridSize, blockSize>>>(WIDTH, HEIGHT, numFrames, states);

    HIP_ASSERT( hipDeviceSynchronize() );
    auto end =  std::chrono::steady_clock::now();
    std::chrono::duration<double> duration = end - start;

    MESSAGE << ( 1.0 / duration.count() ) << " fps\n";

    HIP_ASSERT( hipMemcpy(buffer, deviceBuffer, WIDTH * HEIGHT * 3 * sizeof(unsigned char), hipMemcpyDeviceToHost ) );

}


GLuint createTextureFromBuffer() {
    GLuint textureID;
    glGenTextures(1, &textureID); // Generate a textureID ID
    glBindTexture(GL_TEXTURE_2D, textureID); // Bind the textureID

    // Set textureID parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Upload the buffer data to the textureID
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WIDTH, HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, buffer);
    //glGenerateMipmap(GL_TEXTURE_2D);

    // Unbind the textureID
   // glBindTexture(GL_TEXTURE_2D, 0);

    return textureID; // Return the textureID ID
}


// Shader code (Vertex and Fragment shaders)
const char* vertexShaderSource = R"(
#version 330 core
layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texCoord;
out vec2 TexCoord;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    TexCoord = texCoord;
})";

const char* fragmentShaderSource = R"(
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D texture1;
void main() {
    FragColor = texture(texture1, TexCoord);
})";

GLfloat vertices[] = {
        // Positions         // textureID Coordinates
        -1.f, -1.f,        0.0f, 1.0f,  // Bottom-left
        1.f, -1.f,        1.0f, 1.0f,  // Bottom-right
        -1.f,  1.f,        0.0f, 0.0f,  // Top-left

        1.f, -1.f,        1.0f, 1.0f,  // Bottom-right
        1.f,  1.f,        1.0f, 0.0f,  // Top-right
        -1.f,  1.f,        0.0f, 0.0f   // Top-left
};

GLuint VAO, VBO, textureID;
GLuint shaderProgram;

// Function to compile shaders
GLuint CompileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cerr << "Shader compilation failed: " << infoLog << std::endl;
    }

    return shader;
}

// Function to create and link shaders into a program
GLuint CreateShaderProgram(const char* vertexSource, const char* fragmentSource) {
    GLuint vertexShader = CompileShader(GL_VERTEX_SHADER, vertexSource);
    GLuint fragmentShader = CompileShader(GL_FRAGMENT_SHADER, fragmentSource);

    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        std::cerr << "Shader program linking failed: " << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return program;
}

void InitOpenGL() {
    // Compile shaders and link program
    shaderProgram = CreateShaderProgram(vertexShaderSource, fragmentShaderSource);

    // Set up VBO and VAO
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(0);

    // textureID coordinate attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (GLvoid*)(2 * sizeof(GLfloat)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);

}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
        firstMouse = true;
        pressed = true;
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE) {
        pressed = false;
    }
}

void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos) {
    auto cam = (Camera*)(glfwGetWindowUserPointer(window));
    double sensitivity = 0.005;
    if ( pressed ) {

        if (firstMouse) {
            prevPos = {xpos, ypos};
            firstMouse = false;
        }

        Point2d delta = sensitivity * ( Point2d(xpos, ypos) - prevPos );

        cam->rotateYaw(delta[0]);
        cam->rotatePitch(-delta[1]);

        clearAll();


        prevPos = {xpos, ypos};
    }

}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    auto cam = (Camera*)(glfwGetWindowUserPointer(window));
    constexpr double step = 0.1;

    if ( yoffset < 0 ) {
        if ( cam->focusDistance - step > 0 ) cam->focusDistance -= step;
        cam->init();
        clearAll();
    } else if ( yoffset > 0 ) {
        cam->focusDistance += step;
        cam->init();
        clearAll();
    }

    std::cout << cam->focusDistance << std::endl;
}

void keyboardCallback( GLFWwindow* window, Camera* cam ) {
    constexpr double step = 0.1;

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }

    if ( glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS ) {
        cam->move( { -step, 0, 0 } );
        clearAll();
    }

    if ( glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS ) {
        cam->move( { step, 0, 0 } );
        clearAll();
    }

    if ( glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS ) {
        cam->move( { 0, 0, -step } );
        clearAll();
    }

    if ( glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS ) {
        cam->move( { 0, 0, step } );
        clearAll();
    }

    if ( glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS ) {
        cam->move( { 0, step, 0 } );
        clearAll();
    }

    if ( glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ) {
        cam->move( { 0, -step, 0 } );
        clearAll();
    }

    if ( glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS ) {
        cam->rotateYaw( -step );
        clearAll();
    }

    if ( glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS ) {
        cam->rotateYaw( step );
        clearAll();
    }

    if ( glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS ) {
        cam->rotatePitch( step );
        clearAll();
    }

    if ( glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS ) {
        cam->rotatePitch( -step );
        clearAll();
    }

    if ( glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS ) {
        cam->rotateRoll( step );
        clearAll();
    }

    if ( glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS ) {
        cam->rotateRoll( -step );
        clearAll();
    }

    if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS) {
        if (!keyStateF) {
            keyStateF = true;
            if ( cam->defocusAngle != 0 ) cam->defocusAngle = 0;
            else cam->defocusAngle = oldDefocusAngle;
            cam->init();
            clearAll();
        }
    } else {
        keyStateF = false;
    }
    

}

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

    // Create windowed mode window and OpenGL context
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Realtime raytracer", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwSetCursorPosCallback(window, cursorPositionCallback );
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetScrollCallback(window, scrollCallback);

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, [](GLFWwindow*, int width, int height) {
        glViewport(0, 0, width, height);
    });

    // Initialize GLEW
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    // Initialize OpenGL (Shaders, VBO, VAO, textureID)
    InitOpenGL();

    glViewport( 0, 0, WIDTH * 2, HEIGHT * 2 );

    unsigned char* deviceBuffer;
    HIP_ASSERT( hipMalloc( &deviceBuffer, WIDTH * HEIGHT * 3 * sizeof(unsigned char) ) );

    HIP_ASSERT(  hipMalloc( &memory, WIDTH * HEIGHT * 3 * sizeof(double) ) );

    // Main loop

    auto interval = std::chrono::microseconds(100);

    //fillBuffer( WIDTH, HEIGHT );

    auto lastTime = std::chrono::steady_clock::now();

    auto luaState = Lua::newState();
    if ( !Lua::open( luaState, "/home/auser/dev/src/Collection/test.txt" ) ) {
        std::cerr << "Error loading Lua file: " << lua_tostring(luaState, -1) << "\n";
        Lua::close( luaState );
        return 1;
    }

    Camera* cam = initDeviceCamera( luaState );

    glfwSetWindowUserPointer(window, cam );

    BVH* world = initDeviceWorld( luaState );
    
    world->printDebug();



    hiprandState* states = initStates( WIDTH, HEIGHT );

    while (!glfwWindowShouldClose(window)) {

        clear = false;

        auto currentTime =  std::chrono::steady_clock::now();

        keyboardCallback( window, cam );

        if ( clear ) clearAll();

        if (currentTime - lastTime >= interval) {
            ++numFrames;
            updateBuffer( cam, world, deviceBuffer, states );
            lastTime = currentTime;
        }

//        glBindTexture(GL_TEXTURE_2D, textureID);
        glClearTexImage( textureID, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL );
        textureID = createTextureFromBuffer();
        // Input
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        // Rendering commands
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Use our shader program
        glUseProgram(shaderProgram);

        // Bind textureID
        glBindTexture(GL_TEXTURE_2D, textureID);

        // Render the quad
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);  // Two triangles = 6 vertices
        glBindVertexArray(0);

        // Swap buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    finalize( world, cam, deviceBuffer );

    // Cleanup
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);

    glfwTerminate();
    return 0;
}


// 11.5