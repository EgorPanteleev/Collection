#define FREEGLUT_STATIC
#include <iostream>
#include <GL/freeglut.h>
#include "RayTracer.h"
#include "Scene.h"
#include "Camera.h"
#include "Sphere.h"
#include "Image.h"
int angX = 0;
int angY = 0;
int angZ = 0;
int zoom = 1;
float x1 = 0;
float y1 = 0;
bool key = false;
Camera* cam = new Camera( Vector3f(0,0,0), Vector3f(0,0,1), 1000,500,500 );
Scene* scene = new Scene();
RayTracer rayt( cam, scene );

void read_spc(int k, int, int)
{
//    if (k == GLUT_KEY_HOME)
//    {
//        window.setZoom( -20 );
//    }

    glutPostRedisplay();
}
void read_kb(unsigned char k, int, int)
{
    if (k == 'q')
        angX+=7;
    if (k == 'w')
        angY+=7;

    if (k == 'e')
        angZ+=7;

    if ( k == 'z') {
        x1 -= 30;
        cam->LookAt(Vector3f(x1,y1,1) ,Vector3f(0,1,0));
        key = true;
    }

    if ( k == 'x') {
        x1 += 30;
        cam->LookAt(Vector3f(x1,y1,1) ,Vector3f(0,1,0));
        key = true;
    }

    if ( k == 'c') {
        y1 -= 30;
        cam->LookAt(Vector3f(x1,y1,1) ,Vector3f(0,1,0));
        key = true;
    }

    if ( k == 'd') {
        y1 += 30;
        cam->LookAt(Vector3f(x1,y1,1) ,Vector3f(0,1,0));
        key = true;
    }

    if (k == '+' || k == '=') {
        zoom +=30;
        cam->origin = Vector3f(0,0,zoom);
        key = true;
    }
    if (k == '-' ) {
        zoom -= 30;
        cam->origin = Vector3f(0,0,zoom);
        key = true;
    }
    if (k == '\x1b') {
        delete cam;
        delete scene;
        exit(0);
    }

    glutPostRedisplay();
}
void drawSphere(Vector3f p, float r, int num_segments, RGB color) {
    glPushMatrix();
    glColor3f(color.r, color.g, color.b);
    glTranslatef(p.getX(),p.getY(), p.getZ());
    glutSolidSphere(r,20,20);
    glPopMatrix();
}

bool first = true;
std::vector<Vector3f> fromVec;
std::vector<Vector3f> toVec;
std::vector<Sphere*> spheres;
void RenderScene() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glClearColor(0.8, 0.8, 0.8, 1);
    glLoadIdentity();
//    glTranslatef(0, 0, zoom);
//    glRotatef(angX, 1, 0, 0);
//    glRotatef(angY, 0, 1, 0);
//    glRotatef(angZ, 0, 0, 1);
    glTranslatef(0, 0, -1600);
    glRotatef(0, 1, 0, 0);
    glRotatef(0, 0, 1, 0);
    glRotatef(0, 0, 0, 1);
    double uX = cam->Vx / rayt.getCanvas()->getW();
    double uY = cam->Vy / rayt.getCanvas()->getH();
    if ( first ) {
        first = false;
        spheres.push_back(new Sphere(50, Vector3f(120, 120, 300), RGB(255,0 ,0 )));
        spheres.push_back(new Sphere(20, Vector3f(40, 40, 400), RGB(0, 255, 0)));
        spheres.push_back(new Sphere(30, Vector3f(-20, -20, 500), RGB(0, 0, 255)));
        for ( auto s: spheres ) {
            scene->shapes.push_back( s );
        }
        Light* l = new Light();
        l->origin = Vector3f(-150,300,100);
        l->intensity = 0.8;
        scene->lights.push_back(l);
        rayt.traceAllRays();
    }
    if ( key ) rayt.traceAllRays();
    for ( auto s: spheres ) {
        drawSphere( s->origin, s->radius, 360,s->getColor() );
    }
    glPushMatrix();
    glTranslatef(-cam->Vx / 2,-cam->Vy / 2 ,cam->dV );
    glBegin(GL_QUADS);
    for ( int x = 0; x < rayt.getCanvas()->getW(); ++x ) {
        for ( int y = 0; y < rayt.getCanvas()->getH(); ++y ) {
            RGB color = rayt.getCanvas()->getPixel( x, y );
            glColor3f( color.r / 255, color.g / 255, color.b / 255 );
            glVertex3f(x * uX - uX, y * uY - uY,0 );
            glVertex3f(x * uX, y * uY - uY,0 );
            glVertex3f(x * uX, y * uY,0 );
            glVertex3f(x * uX - uX, y * uY,0 );
        }
    }
    glEnd();
    glPopMatrix();
    glutPostRedisplay();
    glutSwapBuffers();
}

void ReshapeWindow(GLsizei width, GLsizei height) {
    if (height == 0)
        height = 1;
    GLfloat aspect = (GLfloat)width / (GLfloat)height;
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0f, aspect, 0.1f, 1000.0f);
}
int main(int argc, char* argv[]) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(3200, 2560);
    glutCreateWindow("Graphic View");
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glShadeModel(GL_SMOOTH);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glClearColor(0, 0, 0, 1);
    glutKeyboardFunc(read_kb);
    glutDisplayFunc(RenderScene);
    glutReshapeFunc(ReshapeWindow);
    glutSpecialFunc(read_spc);
    glutMainLoop();
    return 0;
}