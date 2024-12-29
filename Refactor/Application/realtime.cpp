
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include "hip/hip_runtime.h"
#include <hip/hip_runtime_api.h>
#include <hip/hip_gl_interop.h>
#include <chrono>
#include "Vector.h"
#include "Sphere.h"
#include "HittableList.h"
#include "Camera.h"
#include "Vec3.h"
#include "RGB.h"
#include "Vec2.h"

const int WIDTH = 800, HEIGHT = 500;
unsigned char* buffer = new unsigned char[WIDTH * HEIGHT * 3];  // RGB buffer

dim3 blockSize(16, 16);
dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (HEIGHT + blockSize.y - 1) / blockSize.y);
bool firstMouse = true;
Point2d prevPos;
Point2d currPos;
Point2d offset;
bool pressed = false;
bool released = false;
double* memory = nullptr;
int cnt = 0;
bool clear;

HOST_DEVICE double linearToGamma( double linear ) {
    if ( linear > 0 ) return std::sqrt( linear );
    return 0;
}

__global__ void clearMemory( int width, int height, double* mem ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ( x >= width || y >= height ) return;

    int idx = ( y * width + x ) * 3;

    mem[idx] = 0;
    mem[idx + 1] = 0;
    mem[idx + 2] = 0;
}

void clearAll( double* memory, int& cnt ) {
    cnt = 0;
    clearMemory<<<gridSize, blockSize>>>( WIDTH, HEIGHT, memory );
}

__global__ void render( Camera* __restrict__ cam, HittableList* __restrict__ world, unsigned char* __restrict__ colorBuffer, double* memory, int cnt,  int width, int height, hiprandState* states )
{

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ( x >= width || y >= height ) return;
//    const double pixelSamplesScale = 1.0 / cam->samplesPerPixel;
    RGB pixelColor = { 0, 0, 0 };
    int idx = y * width + x;
    hiprandState state = states[idx];
    idx *= 3;

    Vec3d pixelCenter = cam->pixel00Loc + (x * cam->pixelDeltaU) + (y * cam->pixelDeltaV);
    Ray ray = cam->getRay( x, y, state );
    pixelColor = cam->traceRay(ray, *world, cam->maxDepth, state );

    const Interval<double> intensity( 0, 1 );

    memory[idx] += intensity.clamp( linearToGamma( pixelColor[0] ) ) * 255;
    memory[idx + 1] += intensity.clamp( linearToGamma( pixelColor[1] ) ) * 255;
    memory[idx + 2] += intensity.clamp( linearToGamma( pixelColor[2] ) ) * 255;


    colorBuffer[idx] = static_cast<unsigned char>( memory[idx] / cnt );
    colorBuffer[idx + 1] = static_cast<unsigned char>( memory[idx + 1] / cnt );
    colorBuffer[idx + 2] = static_cast<unsigned char>( memory[idx + 2] / cnt );

}

__global__ void render_init(int max_x, int max_y, int seed, hiprandState* states ) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    hiprand_init(seed, pixel_index, 0, &states[pixel_index] );
}

HittableList* initDeviceWorld() {
    Lambertian* ground = new Lambertian( { 0.4, 0.4, 0.4 } );
    Lambertian* center = new Lambertian( { 0.1, 0.2, 0.5 } );
    Dielectric* left = new Dielectric( 1.5 );
    Dielectric* bubble = new Dielectric( 1.0 / 1.5 );
    Metal* right = new Metal( { 0.8, 0.6, 0.2 }, 0.15 );

    HittableList world;

    world.add( new Sphere( 10000, { 0, -10000.5, -1 }, ground ) );
    world.add( new Sphere( 0.5, { 0, 0, -1.2 }, center ) );
    world.add( new Sphere( 0.5, { -1, 0, -1 }, left ) );
    world.add( new Sphere( 0.4, { -1, 0, -1 }, bubble ) );
    world.add( new Sphere( 0.5, { 1, 0, -1 }, right ) );

    auto worldDevice = world.copyToDevice();
    return worldDevice;
}

Camera* initDeviceCamera() {
    Camera cam;
    cam.aspectRatio = 16.0 / 10.0;
    cam.imageWidth = WIDTH;
    cam.samplesPerPixel = 1;
    cam.maxDepth = 10;
    cam.vFOV = 30;

    cam.lookFrom = { 0, 0, 1 };
    cam.lookAt = { 0, 0, -1 };
    cam.up = { 0, 1, 0 };

    cam.init();
    auto deviceCamera = HIP::allocateOnDevice<Camera>();

    HIP::copyToDevice( &cam, deviceCamera );
    return deviceCamera;
}

void finalize( HittableList* world, Camera* cam, unsigned char* deviceBuffer, double* memory ) {
    world->deallocateOnDevice();
    HIP::deallocateOnDevice( cam );
    HIP::deallocateOnDevice( deviceBuffer );
    HIP::deallocateOnDevice( memory );
}

unsigned char* initDeviceBuffer( int width, int height, int channels ) {
    return HIP::allocateOnDevice<unsigned char>( width * height * channels * sizeof( unsigned char ) );
}

hiprandState* initStates( int width, int height ) {
    hiprandState *states;
    HIP_ASSERT(hipMalloc((void **)&states, width * height * sizeof(hiprandState)));
    render_init<<<gridSize, blockSize>>>(width, height, 1984, states);
    return states;
}


__global__ void fillBufferKernel( int width, int height, unsigned char* deviceBuffer) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if((x >= width) || (y >= height)) return;
    int index = (y * width + x) * 3;
    unsigned char color = static_cast<unsigned char>((255 * x) / (width - 1));

    // Set R, G, B channels to the same gradient value
    deviceBuffer[index] = color;     // Red channel
    deviceBuffer[index + 1] = color; // Green channel
    deviceBuffer[index + 2] = color; // Blue channel

}



void fillBuffer( int width, int height ) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int index = (y * width + x) * 3;

            // Calculate the gradient value for the current pixel
            unsigned char color = static_cast<unsigned char>((255 * x) / (width - 1));

            // Set R, G, B channels to the same gradient value
            buffer[index] = color;     // Red channel
            buffer[index + 1] = color; // Green channel
            buffer[index + 2] = color; // Blue channel
        }
    }
}

void updateBuffer( int width, int height, Camera* cam, HittableList* world, double* memory, int cnt, unsigned char* deviceBuffer, hiprandState* states ) {
    render<<<gridSize, blockSize>>>( cam, world, deviceBuffer, memory, cnt, width, height, states );
    render_init<<<gridSize, blockSize>>>(width, height, cnt, states);
//    fillBufferKernel<<< gridSize, blockSize>>>
//    ( WIDTH, HEIGHT, deviceBuffer );
    hipDeviceSynchronize();
    hipMemcpy(buffer, deviceBuffer, width * height * 3 * sizeof(unsigned char), hipMemcpyDeviceToHost );
    //hipMemcpy(device, host, n * sizeof(Type), hipMemcpyHostToDevice

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
//    glGenerateMipmap(GL_TEXTURE_2D);

    // Unbind the textureID
    glBindTexture(GL_TEXTURE_2D, 0);

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
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        firstMouse = true;
        pressed = true;
    }
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
        pressed = false;
    }
}

void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos) {
    auto cam = (Camera*)(glfwGetWindowUserPointer(window));
    double sensetivity = 0.005;
    if ( pressed ) {

        if (firstMouse) {
            prevPos = {xpos, ypos};
            firstMouse = false;
        }

        Point2d delta = sensetivity * ( Point2d(xpos, ypos) - prevPos );

        cam->rotateYaw(delta[0]);
        cam->rotatePitch(delta[1]);

        clearAll( memory, cnt );

        //TODO make normal camera and interface for movement and etc

        prevPos = {xpos, ypos};
    }

}

void mouseCallback( GLFWwindow* window, Camera* cam ) {

//    if ( offset[0] != 0 ) {
//        cam->rotateYaw( offset[0] );
//        clear = true;
//    }
//
//    if ( offset[1] != 0 ) {
//        cam->rotatePitch( offset[1] );
//        clear = true;
//    }
//
//    if ( pressed )  {
//        //prevPos = currPos;
//        offset = {};
//    }

}

void keyboardCallback( GLFWwindow* window, Camera* cam ) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }

    if ( glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS ) {
        cam->move( { -0.1, 0, 0 } );
        clear = true;
    }

    if ( glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS ) {
        cam->move( { 0.1, 0, 0 } );
        clear = true;
    }

    if ( glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS ) {
        cam->move( { 0, 0, -0.1 } );
        clear = true;
    }

    if ( glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS ) {
        cam->move( { 0, 0, 0.1 } );
        clear = true;
    }

    if ( glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS ) {
        cam->move( { 0, 0.1, 0 } );
        clear = true;
    }

    if ( glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ) {
        cam->move( { 0, -0.1, 0 } );
        clear = true;
    }

    if ( glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS ) {
        cam->rotateYaw( -0.1 );
        clear = true;
    }

    if ( glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS ) {
        cam->rotateYaw( 0.1 );
        clear = true;
    }

    if ( glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS ) {
        cam->rotatePitch( 0.1 );
        clear = true;
    }

    if ( glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS ) {
        cam->rotatePitch( -0.1 );
        cam->init();
        clear = true;
    }

    if ( glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS ) {
        cam->rotateRoll( 0.1 );
        cam->init();
        clear = true;
    }

    if ( glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS ) {
        cam->rotateRoll( -0.1 );
        cam->init();
        clear = true;
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
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "OpenGL Quad with textureID", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwSetCursorPosCallback(window, cursorPositionCallback );
    glfwSetMouseButtonCallback(window, mouseButtonCallback);

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
    hipMalloc( &deviceBuffer, WIDTH * HEIGHT * 3 * sizeof(unsigned char) );

    hipMalloc( &memory, WIDTH * HEIGHT * 3 * sizeof(double) );

    // Main loop

    auto interval = std::chrono::microseconds(100);

    //fillBuffer( WIDTH, HEIGHT );

    auto lastTime = std::chrono::steady_clock::now();

    Camera* cam = initDeviceCamera();

    glfwSetWindowUserPointer(window, cam );

    HittableList* world = initDeviceWorld();

    hiprandState* states = initStates( WIDTH, HEIGHT );

    while (!glfwWindowShouldClose(window)) {

        clear = false;

        auto currentTime =  std::chrono::steady_clock::now();

        keyboardCallback( window, cam );

        mouseCallback( window, cam );

        if ( clear ) clearAll( memory, cnt );

        if (currentTime - lastTime >= interval) {
            updateBuffer( WIDTH, HEIGHT, cam, world, memory, ++cnt, deviceBuffer, states );
            lastTime = currentTime;
        }

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

    finalize( world, cam, deviceBuffer, memory );

    // Cleanup
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);

    glfwTerminate();
    return 0;
}
