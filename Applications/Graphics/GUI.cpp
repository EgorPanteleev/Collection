#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include "RayTracer.h"
#include "Scene.h"
#include "Camera.h"
#include "SphereMesh.h"
#include "Image.h"
#include <ctime>
#include "CubeMesh.h"
#include "BaseMesh.h"
#include "TriangularMesh.h"
#include "Material.h"
#include "PointLight.h"
#include "SpotLight.h"
#include "cstdlib"
//#include "GroupOfMeshes.h"
#include "Denoiser.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
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


GLuint LoadTextureFromFile(const char* filename, int* width, int* height)
{
    int nrChannels;
    unsigned char* data = stbi_load(filename, width, height, &nrChannels, 0);
    if (!data)
    {
        std::cerr << "Failed to load texture" << std::endl;
        return 0;
    }

    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    GLenum format = (nrChannels == 4) ? GL_RGBA : GL_RGB;
    glTexImage2D(GL_TEXTURE_2D, 0, format, *width, *height, 0, format, GL_UNSIGNED_BYTE, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT); // Set texture wrapping to GL_REPEAT
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // Use linear filtering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    stbi_image_free(data);

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
            RGB color = canvas->getPixel( x, h - y );
            data[(y * w + x) * 3 + 0] = color.r;
            data[(y * w + x) * 3 + 1] = color.g;
            data[(y * w + x) * 3 + 2] = color.b;
        }
    }

    // Create texture from rasterized data
    return createTexture(data, w, h);
}
void loadScene( Scene* scene, std::vector <BaseMesh*>& meshes, std::vector<Light*>& lights ) {
    for ( const auto& mesh: meshes ) {
        scene->meshes.push_back( mesh );
    }
    for ( const auto& light: lights ) {
        scene->lights.push_back( light );
    }
}

void testScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    Camera* cam = new Camera( Vector3f(0,10,0 ), Vector3f(0,0,1), 2400,3200,2000 );
    //Camera* cam = new Camera( Vector3f(0,0,0 ), Vector3f(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );

    std::vector<BaseMesh*> meshes;
    std::vector<Light*> lights;
    float roomRefl = 0;
////right
    meshes.push_back( new CubeMesh( Vector3f(70, -50, 0), Vector3f(80, 70, 600),
                                    { GREEN, -1 , roomRefl } ) );
////left
    meshes.push_back(new CubeMesh( Vector3f(-80, -50, 0), Vector3f(-70, 70, 600),
                                   { RED, -1 , roomRefl } ) );
////front
    meshes.push_back(new CubeMesh( Vector3f(-100, -50, 290), Vector3f(100, 70, 300),
                                   { GRAY, -1 , roomRefl } ) );
////back
    meshes.push_back(new CubeMesh( Vector3f(-100, -50, -10), Vector3f(100, 70, 0),
                                   { GRAY, -1 , roomRefl } ) );
////down
    meshes.push_back(new CubeMesh( Vector3f(-100, -70, 0), Vector3f(100, -50, 620),
                                   { GRAY, -1 , roomRefl } ) );
////up
    meshes.push_back(new CubeMesh( Vector3f(-100, 70, roomRefl), Vector3f(100, 90, 620),
                                   { GRAY, -1 , 0 } ) );

////RAND BLOCK
    auto* randBlockForward = new CubeMesh( Vector3f(-15, -50, 310), Vector3f(15, -30, 340) );
    randBlockForward->moveTo( Vector3f(0, -40, 325) );
    randBlockForward->scaleTo( Vector3f(30,100,30) );
    randBlockForward->rotate( Vector3f( 0,1,0), 25);
    randBlockForward->move( Vector3f(30,0,0));
    randBlockForward->setMaterial({GRAY, -1 , 0});
    randBlockForward->move( Vector3f(-10,0,-150));
    //randBlockForward->scaleTo( 200 );
    meshes.push_back(randBlockForward );

    auto* randBlockForward2 = new CubeMesh( Vector3f(-15, -50, 310), Vector3f(15, -30, 340) );
    randBlockForward2->moveTo( Vector3f(0, -40, 325) );
    randBlockForward2->scaleTo( Vector3f(30,260,30) );
    randBlockForward2->rotate( Vector3f( 0,1,0), -25);
    randBlockForward2->move( Vector3f(35,0,0));
    randBlockForward2->setMaterial({GRAY, -1 , 0.8});
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
}


void saveToPNG( Canvas* canvas, const std::string& fileName ) {
    // Create an array to store pixel data (RGBA format)
    unsigned char* image = new unsigned char[canvas->getW() * canvas->getH() * 4];
    for (int y = 0; y < canvas->getH(); ++y) {
        for (int x = 0; x < canvas->getW(); ++x) {
            int index = (y * canvas->getW() + x) * 4;
            image[index + 0] = canvas->getPixel( x, canvas->getH() - 1 - y  ).r;
            image[index + 1] = canvas->getPixel( x, canvas->getH() - 1 - y ).g;
            image[index + 2] = canvas->getPixel( x, canvas->getH() - 1 - y ).b;
            image[index + 3] = 255;  // Alpha component (opaque)
        }
    }

    // Save the image as a PNG file
    if (stbi_write_png( fileName.c_str(), canvas->getW(), canvas->getH(), 4, image, canvas->getW() * 4))
        std::cout << "Image saved successfully: " << fileName << std::endl;
    else std::cerr << "Failed to save image: " << fileName << std::endl;

    // Clean up
    delete[] image;
}


int main(  int argc, char* argv[]  )
{
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

    // Load an image
    int image_width = 0, image_height = 0;
    GLuint myTexture = LoadTextureFromFile("", &image_width, &image_height);
//    if (myImageTexture == 0)
//    {
//        std::cerr << "Failed to load texture" << std::endl;
//        return -1;
//    }
    // Set the scale factor

    // Main loop
    Kokkos::initialize(argc, argv); {
    RayTracer* rayTracer = nullptr;
    //                int w = io.DisplaySize.x;
    //                int h = io.DisplaySize.y;
    int w = 1920;
    int h = 1200;
    int depth = 2;
    int ambientSamples = 1;
    int lightSamples = 1;
    testScene( rayTracer, w, h, depth, ambientSamples, lightSamples );

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
            setenv("OMP_PROC_BIND", "spread", 1);
            setenv("OMP_PLACES", "threads", 1);
                srand(time( nullptr ));
                auto start = std::chrono::high_resolution_clock::now();
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> loadTime = end - start;
                std::cout << "Model loads "<< loadTime.count() << " seconds" << std::endl;
                start = std::chrono::high_resolution_clock::now();;
                rayTracer->traceAllRays( RayTracer::PARALLEL );
                end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> renderTime = end - start;
                std::cout << "RayTracer works "<< renderTime.count() << " seconds" << std::endl;
        }
        if ( ImGui::Button("Denoise")) {
            Denoiser::denoise( rayTracer->getCanvas()->getData(), rayTracer->getCanvas()->getW(), rayTracer->getCanvas()->getH() );
        }

        //ImGui::Separator();
        ImGui::End();
        // Second window with an image
        ImGui::Begin("Image", nullptr, ImGuiWindowFlags_NoDecoration  | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar );
        ImGui::Image((void*)(intptr_t)getTexture( rayTracer->getCanvas() ), ImVec2( ImGui::GetWindowSize().x , ImGui::GetWindowSize().y ));
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