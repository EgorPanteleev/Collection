
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include "hip/hip_runtime.h"
#include <hip/hip_runtime_api.h>
#include <hip/hip_gl_interop.h>
#include <chrono>

const int WIDTH = 960, HEIGHT = 600;
unsigned char* buffer = new unsigned char[WIDTH * HEIGHT * 3];  // RGB buffer

dim3 blockSize(16, 16);
dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (HEIGHT + blockSize.y - 1) / blockSize.y);

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

void updateBuffer( int width, int height, unsigned char* deviceBuffer ) {
    fillBufferKernel<<< gridSize, blockSize>>>
    ( WIDTH, HEIGHT, deviceBuffer );
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
        -1.f, -1.f,        0.0f, 0.0f,  // Bottom-left
        1.f, -1.f,        1.0f, 0.0f,  // Bottom-right
        -1.f,  1.f,        0.0f, 1.0f,  // Top-left

        1.f, -1.f,        1.0f, 0.0f,  // Bottom-right
        1.f,  1.f,        1.0f, 1.0f,  // Top-right
        -1.f,  1.f,        0.0f, 1.0f   // Top-left
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

    // Main loop

    auto interval = std::chrono::microseconds(100);

    //fillBuffer( WIDTH, HEIGHT );

    auto lastTime = std::chrono::steady_clock::now();

    while (!glfwWindowShouldClose(window)) {

        auto currentTime =  std::chrono::steady_clock::now();
        if (currentTime - lastTime >= interval) {
            updateBuffer( WIDTH, HEIGHT, deviceBuffer );
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

    hipFree( deviceBuffer );

    // Cleanup
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);

    glfwTerminate();
    return 0;
}
