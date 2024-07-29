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
//#include "Denoiser.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
// Function to load a texture from a file using stb_image
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
void loadScene( Scene* scene, std::vector <BaseMesh*>& meshes, std::vector<Light*>& lights ) {
    for ( const auto& mesh: meshes ) {
        scene->meshes.push_back( mesh );
    }
    for ( const auto& light: lights ) {
        scene->lights.push_back( light );
    }
}

void testScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    float FOV = 100;
    float dV = w / 2 / tan( FOV * M_PI / 360  );
//    Camera* cam = new Camera( Vector3f(-4.24462,-0.129327,1.31629 ), Vector3f(0,0,1), dV,w,h );
    Camera* cam = new Camera( Vector3f(-1,0,-3 ), Vector3f(0.3,0,1), dV,w,h );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );
    std::vector<BaseMesh*> meshes;
    std::vector<Light*> lights;

    auto* room = new TriangularMesh();
    room->loadMesh( "/home/auser/dev/src/Collection/Models/restroom/restroom.obj" );
//    room->rotate( Vector3f( 0, 0, 1), 30 );
    room->rotate( Vector3f( 0,1,0),90);
//    room->rotate( Vector3f( 1,0,0),90);
//    room->move( Vector3f( 0,0,100) );
    room->setMaterial( { RGB( 130, 130, 130 ), 1 , 0, 1 } );
    meshes.push_back( room );


    lights.push_back( new PointLight( Vector3f(0 ,1.5,-1), 3));
//    lights.push_back( new PointLight( cam->origin, 1 ) );
//    float lightWidth = 0.3;
//    float lightLength = 0.3;
//    int roomHeight = 4;
//    int roomLength = 5;
//    lights.push_back( new SpotLight( Vector3f(0 - lightWidth,roomHeight/2,roomLength/2 - lightLength),
//                                     Vector3f(0 + lightWidth,roomHeight/2,roomLength/2 + lightLength), 1));
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
    GLuint myImageTexture = LoadTextureFromFile("", &image_width, &image_height);
//    if (myImageTexture == 0)
//    {
//        std::cerr << "Failed to load texture" << std::endl;
//        return -1;
//    }
    // Set the scale factor

    // Main loop
    RayTracer* rayTracer = nullptr;
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
            Kokkos::initialize(argc, argv); {
                srand(time( nullptr ));
                int w = io.DisplaySize.x; ;
                int h = io.DisplaySize.y;
                int depth = 2;
                int ambientSamples = 1;
                int lightSamples = 5;
                auto start = std::chrono::high_resolution_clock::now();
                testScene( rayTracer, w, h, depth, ambientSamples, lightSamples );
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> loadTime = end - start;
                std::cout << "Model loads "<< loadTime.count() << " seconds" << std::endl;
                start = std::chrono::high_resolution_clock::now();;
                rayTracer->traceAllRays( RayTracer::PARALLEL );
                end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> renderTime = end - start;
                std::cout << "RayTracer works "<< renderTime.count() << " seconds" << std::endl;
                saveToPNG( rayTracer->getCanvas(), "out.png" );
            } Kokkos::finalize();
            myImageTexture = LoadTextureFromFile("/home/auser/dev/src/Collection/release/Applications/Graphics/out.png", &image_width, &image_height);
        }
        if ( ImGui::Button("Denoise")) {
            Denoiser::denoise( rayTracer->getCanvas()->getData(), rayTracer->getCanvas()->getW(), rayTracer->getCanvas()->getH() );
            saveToPNG( rayTracer->getCanvas(), "outDenoised.png" );
            myImageTexture = LoadTextureFromFile("/home/auser/dev/src/Collection/release/Applications/Graphics/outDenoised.png", &image_width, &image_height);
        }

        //ImGui::Separator();
        ImGui::End();
        // Second window with an image
        ImGui::Begin("Image", nullptr, ImGuiWindowFlags_NoDecoration  | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar );
        ImGui::Image((void*)(intptr_t)myImageTexture, ImVec2( ImGui::GetWindowSize().x , ImGui::GetWindowSize().y ));
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

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}