#define FREEGLUT_STATIC
#include <iostream>
#include <GL/freeglut.h>
#include "RayTracer.h"
#include "Scene.h"
#include "Camera.h"
#include "Shape.h"
#include "Image.h"
#include "Object.h"
#include "Light.h"
#include "Cube.h"
#define GRAY RGB( 210, 210, 210 )
#define RED RGB( 255, 0, 0 )
#define GREEN RGB( 0, 255, 0 )
#define BLUE RGB( 0, 0, 255 )
#define YELLOW RGB( 255, 255, 0 )
#define BROWN RGB( 150, 75, 0 )
#define PINK RGB( 255,105,180 )
#define DARK_BLUE RGB(65,105,225)
int angX = 0;
int angY = 0;
int angZ = 0;
int zoom = 1;
float x1 = 0;
float y1 = 0;
bool key = false;
Camera* cam = new Camera( Vector3f(0, 0,0), Vector3f(0,0,1), 600 * 1.9,320* 1.9,200* 1.9 );
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
        x1 -= 10;
        cam->LookAt(Vector3f(x1,y1,100) ,Vector3f(0,1,0));
        key = true;
    }

    if ( k == 'x') {
        x1 += 10;
        cam->LookAt(Vector3f(x1,y1,100) ,Vector3f(0,1,0));
        key = true;
    }

    if ( k == 'c') {
        y1 -= 10;
        cam->LookAt(Vector3f(x1,y1,100) ,Vector3f(0,1,0));
        key = true;
    }

    if ( k == 'd') {
        y1 += 10;
        cam->LookAt(Vector3f(x1,y1,100) ,Vector3f(0,1,0));
        key = true;
    }

    if (k == '+' || k == '=') {
        zoom +=10;
        cam->origin = Vector3f(0,0,zoom);
        key = true;
    }
    if (k == '-' ) {
        zoom -= 10;
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
std::vector<Shape*> shapes;
std::vector<Material> materials;
std::vector<Light*> ligths;
void RenderScene() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glClearColor(0.8, 0.8, 0.8, 1);
    glLoadIdentity();
    glTranslatef(0, 0, -1600);
    glRotatef(0, 1, 0, 0);
    glRotatef(0, 0, 1, 0);
    glRotatef(0, 0, 0, 1);
    double uX = cam->Vx / rayt.getCanvas()->getW();
    double uY = cam->Vy / rayt.getCanvas()->getH();
    if ( first ) {
        first = false;
        auto* randBlockForward = new Cube( Vector3f(-15, -50, 310), Vector3f(15, -30, 340) );
        //randBlock->move( Vector3f(-30,0,0 ));
        //randBlock->rotate( Vector3f( 0,1,0), 45);
        shapes.push_back(randBlockForward );
        materials.emplace_back( GRAY, 1 , 0 );

        auto* randBlockBackward = new Cube( Vector3f(15, -50, -310), Vector3f(-15, -30, -340) );
        //randBlock->move( Vector3f(-30,0,0 ));
        //randBlock->rotate( Vector3f( 0,1,0), 45);
        shapes.push_back(randBlockBackward );
        materials.emplace_back( GRAY, 1 , 0 );

        auto* randBlockLeft = new Cube( Vector3f(-300, -30, -15), Vector3f(-340, -50, 15) );
        //randBlock->move( Vector3f(-30,0,0 ));
        //randBlock->rotate( Vector3f( 0,1,0), 45);
        shapes.push_back(randBlockLeft );
        materials.emplace_back( GRAY, 1 , 0 );

        auto* randBlockRight = new Cube( Vector3f(300, -50, -15), Vector3f(340, -30, 15) );
        //randBlock->move( Vector3f(-30,0,0 ));
        //randBlock->rotate( Vector3f( 0,1,0), 45);
        shapes.push_back(randBlockRight );
        materials.emplace_back( GRAY, 1 , 0 );

        auto* randBlockUp = new Cube( Vector3f(-15, 300, -15), Vector3f(15, 340, 15) );
        //randBlock->move( Vector3f(-30,0,0 ));
        //randBlock->rotate( Vector3f( 0,1,0), 45);
        shapes.push_back(randBlockUp );
        materials.emplace_back( GRAY, 1 , 0 );

        auto* randBlockDown = new Cube( Vector3f(15, -300, -15), Vector3f(-15, -340, 15) );
        //randBlock->move( Vector3f(-30,0,0 ));
        //randBlock->rotate( Vector3f( 0,1,0), 45);
        shapes.push_back( randBlockDown );
        materials.emplace_back( GRAY, 1 , 0 );

        ligths.push_back( new Light( Vector3f(0,45,300), 0.6));
        ligths.push_back( new Light( Vector3f(0,0,-900), 0.6));
        ligths.push_back( new Light( Vector3f(0,0,900), 0.6));

        for ( int i = 0; i < shapes.size(); ++i ) {
            scene->objects.push_back( new Object( shapes[i], materials[i] ) );
        }
        for ( auto l: ligths ) {
            scene->lights.push_back( l );
        }
        rayt.traceAllRaysWithThreads( 40 );
    }
    if ( key ) rayt.traceAllRaysWithThreads( 40 );
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