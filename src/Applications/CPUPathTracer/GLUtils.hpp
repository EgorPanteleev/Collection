//
// Created by igor on 10/17/25.
//

#ifndef GLUTILS_HPP
#define GLUTILS_HPP

#include <GL/glew.h>
#include <GLFW/glfw3.h>

GLuint createShaderProgram();

bool initGLEW();

GLuint createTexture(int width, int height, void* data);

void updateTexture(GLuint tex, int width, int height, void* data);

void createBuffers(GLuint& VAO, GLuint& VBO, GLuint& EBO);

void drawTexture(GLuint shader, GLuint VAO, GLuint tex);

void cleanData(GLuint tex, GLuint shader, GLuint VBO, GLuint EBO, GLuint VAO);

#endif //GLUTILS_HPP
