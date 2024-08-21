#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include "RayTracer.h"
#include "Scene.h"
#include "Camera.h"
#include "Sphere.h"
#include <ctime>
#include "CubeMesh.h"
#include "Mesh.h"
#include "Material.h"
#include "PointLight.h"
#include "SpotLight.h"
#include "cstdlib"
#include "Rasterizer.h"
//#include "GroupOfMeshes.h"
#include "Denoiser.h"
#define GRAY RGB( 210, 210, 210 )
#define RED RGB( 255, 0, 0 )
#define GREEN RGB( 0, 255, 0 )
#define BLUE RGB( 0, 0, 255 )
#define YELLOW RGB( 255, 255, 0 )
#define BROWN RGB( 150, 75, 0 )
#define PINK RGB( 255,105,180 )
#define DARK_BLUE RGB(65,105,225)
#define CYAN RGB( 0, 255, 255)
// Function to load a texture from a file using stb_image

GLuint createTexture(const unsigned char* data, int width, int height) {
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);
    return textureID;
}

GLuint getTexture( Canvas* canvas ) {
    // Rasterized data (example)
    int w = canvas->getW();
    int h = canvas->getH();
    unsigned char* data = new unsigned char[w * h * 3];
    // Fill the rasterized data with a simple gradient for demonstration
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            RGB color = canvas->getColor( x, h - y );
            data[(y * w + x) * 3 + 0] = color.r;
            data[(y * w + x) * 3 + 1] = color.g;
            data[(y * w + x) * 3 + 2] = color.b;
        }
    }

    // Create texture from rasterized data
    return createTexture(data, w, h);
}
void loadScene(Scene* scene, Vector <Mesh*>& meshes, Vector<Light*>& lights ) {
    for ( const auto& mesh: meshes ) {
        scene->add( mesh );
    }
    for ( const auto& light: lights ) {
        scene->add( light );
    }
}

void testScene( RayTracer*& rayTracer, Rasterizer*& rasterizer, int w, int h, int d, int numAS, int numLS ) {
    Camera* cam = new Camera( Vector3f(0,10,0 ), Vector3f(0,0,1), 2400,3200,2000 );
    //Camera* cam = new Camera( Vector3f(0,0,0 ), Vector3f(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );

    Vector<Mesh*> meshes;
    Vector<Light*> lights;
    float roomRefl = 0;
////right
    meshes.push_back( new CubeMesh( Vector3f(70, -50, 0), Vector3f(80, 70, 290),
                                    { GREEN, -1 , roomRefl } ) );
////left
    meshes.push_back(new CubeMesh( Vector3f(-80, -50, 0), Vector3f(-70, 70, 290),
                                   { RED, -1 , roomRefl } ) );
////front
    meshes.push_back(new CubeMesh( Vector3f(-100, -50, 290), Vector3f(100, 70, 300),
                                   { GRAY, -1 , roomRefl } ) );
////back
    meshes.push_back(new CubeMesh( Vector3f(-100, -50, -10), Vector3f(100, 70, 0),
                                   { GRAY, -1 , roomRefl } ) );
////down
    meshes.push_back(new CubeMesh( Vector3f(-100, -70, 0), Vector3f(100, -50, 300),
                                   { GRAY, -1 , roomRefl } ) );
////up
    meshes.push_back(new CubeMesh( Vector3f(-100, 70, roomRefl), Vector3f(100, 90, 300),
                                   { GRAY, -1 , 0 } ) );

////RAND BLOCK
    auto* randBlockForward = new CubeMesh( Vector3f(-15, -50, 310), Vector3f(15, -30, 340) );
    randBlockForward->moveTo( Vector3f(0, -40, 325) );
    randBlockForward->scaleTo( Vector3f(30,100,30) );
    randBlockForward->rotate( Vector3f( 0,1,0), 25);
    randBlockForward->move( Vector3f(30,0,0));
    randBlockForward->setMaterial({CYAN, -1 , 0});
    randBlockForward->move( Vector3f(-10,0,-150));
    //randBlockForward->scaleTo( 200 );
    meshes.push_back(randBlockForward );

    auto* randBlockForward2 = new CubeMesh( Vector3f(-15, -50, 310), Vector3f(15, -30, 340) );
    randBlockForward2->moveTo( Vector3f(0, -40, 325) );
    randBlockForward2->scaleTo( Vector3f(30,260,30) );
    randBlockForward2->rotate( Vector3f( 0,1,0), -25);
    randBlockForward2->move( Vector3f(35,0,0));
    randBlockForward2->setMaterial({PINK, -1 , 0.8});
    randBlockForward2->move( Vector3f(-50,0,-100));
    //randBlockForward2->scaleTo( 200 );
    meshes.push_back(randBlockForward2 );

////LIGHTS

    //lights.push_back( new PointLight( Vector3f(0,65,150), 0.55));
    int lightWidth = 20;
    lights.push_back( new SpotLight( Vector3f(0 - lightWidth,65,180 - lightWidth), Vector3f(0 + lightWidth,65,180 + lightWidth), 0.7));

////LOADING...
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
    rasterizer = new Rasterizer( cam, scene, canvas );
}
int main(  int argc, char* argv[]  )
{
    srand(time( nullptr ));
    setenv("OMP_PROC_BIND", "spread", 1);
    setenv("OMP_PLACES", "threads", 1);
    // Initialize GLFW
    if (!glfwInit())
        return -1;
    glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, 1);
    //glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
    glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);
    // Create a windowed mode window and its OpenGL context
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Hello, ImGui!", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync


    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_::ImGuiConfigFlags_DockingEnable;
    // io.ConfigFlags |= ImGuiConfigFlags_::ImGuiConfigFlags_ViewportsEnable;

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 460");

    // Set the scale factor

    // Main loop
    Kokkos::initialize(argc, argv); {
    RayTracer* rayTracer = nullptr;
    Rasterizer* rasterizer = nullptr;
    //                int w = io.DisplaySize.x;
    //                int h = io.DisplaySize.y;
    int w = 3200;
    int h = 2000;
    int depth = 1;
    int ambientSamples = 1;
    int lightSamples = 1;
    Canvas* textureCanvas = nullptr;
    testScene( rayTracer, rasterizer, w, h, depth, ambientSamples, lightSamples );

    GLuint texture;

    while (!glfwWindowShouldClose(window))
    {
        // Poll and handle events
        glfwPollEvents();

        // Start the ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
/// ---///////////////////
        ImGui::DockSpaceOverViewport( 0, ImGui::GetMainViewport());
        //ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());
        // First window with buttons
        ImGui::Begin("Control Window", nullptr, ImGuiWindowFlags_NoDecoration  | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar);
        ImGui::Text("Button Controls:");
        if (ImGui::Button("Ray Tracer")) {
            auto start = std::chrono::high_resolution_clock::now();;
            rayTracer->render( RayTracer::PARALLEL );
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> renderTime = end - start;
            std::cout << "RayTracer works "<< renderTime.count() << " seconds" << std::endl;
            textureCanvas = rayTracer->getCanvas();
            texture = getTexture( rayTracer->getCanvas() );
        }
        if (ImGui::Button("Rasterizator")) {
            auto start = std::chrono::high_resolution_clock::now();;
            rasterizer->render();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> renderTime = end - start;
            std::cout << "Rasterizator works "<< renderTime.count() << " seconds" << std::endl;
            textureCanvas = rasterizer->getCanvas();
            texture = getTexture( rasterizer->getCanvas() );
        }

        if ( ImGui::Button("Denoise")) {
            if ( textureCanvas != nullptr )
                Denoiser::denoise( textureCanvas->getColorData(), textureCanvas->getNormalData(), textureCanvas->getAlbedoData(), rayTracer->getCanvas()->getW(), rayTracer->getCanvas()->getH() );
            texture = getTexture( textureCanvas );
        }

        //ImGui::Separator();
        ImGui::End();
        // Second window with an image
        ImGui::Begin("Image", nullptr, ImGuiWindowFlags_NoDecoration  | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar );
        ImGui::Image((void*)(intptr_t)texture, ImVec2( ImGui::GetWindowSize().x , ImGui::GetWindowSize().y ));
        ImGui::End();
/// ---///////////////////
        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
//        if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
//        {
//            GLFWwindow* backup_current_context = glfwGetCurrentContext();
//            ImGui::UpdatePlatformWindows();
//            ImGui::RenderPlatformWindowsDefault();
//            glfwMakeContextCurrent(backup_current_context);
//        }

        // Swap buffers
        glfwSwapBuffers(window);
    }
    } Kokkos::finalize();
    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}