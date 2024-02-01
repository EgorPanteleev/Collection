#include <iostream>

#include <fstream>

#include "OBJ_Loader.h"
#include "Shape.h"
//C:/Users/igor/CLionProjects/Collection/OBJReader/
int main() {
    objl::Loader loader;
    bool a = loader.LoadFile( "C:/Users/igor/CLionProjects/Collection/OBJReader/model.obj");
    if ( !a ) return -1;
    std::ofstream file("outMeshes.txt");
    // Go through each loaded mesh and out its contents
    for (int i = 0; i < loader.LoadedMeshes.size(); i++) {
        // Copy one of the loaded meshes to be our current mesh
        objl::Mesh curMesh = loader.LoadedMeshes[i];

        // Print Mesh Name
        file << "Mesh " << i << ": " << curMesh.MeshName << "\n";

        // Print Vertices
        file << "Vertices:\n";

        // Go through each vertex and print its number,
        //  position, normal, and texture coordinate
        for (int j = 0; j < curMesh.Vertices.size(); j++) {
            file << "V" << j << ": " <<
                 "P(" << curMesh.Vertices[j].Position.X << ", " << curMesh.Vertices[j].Position.Y << ", " << curMesh.Vertices[j].Position.Z << ") " <<
                 "N(" << curMesh.Vertices[j].Normal.X << ", " << curMesh.Vertices[j].Normal.Y << ", " << curMesh.Vertices[j].Normal.Z << ") " <<
                 "TC(" << curMesh.Vertices[j].TextureCoordinate.X << ", " << curMesh.Vertices[j].TextureCoordinate.Y << ")\n";
        }

        // Print Indices
        file << "Indices:\n";

        // Go through every 3rd index and print the
        //	triangle that these indices represent
        for (int j = 0; j < curMesh.Indices.size(); j += 3) {
            file << "T" << j / 3 << ": " << curMesh.Indices[j] << ", " << curMesh.Indices[j + 1] << ", " << curMesh.Indices[j + 2] << "\n";
        }

        // Print Material
        file << "Material: " << curMesh.MeshMaterial.name << "\n";
        file << "Ambient Color: " << curMesh.MeshMaterial.Ka.X << ", " << curMesh.MeshMaterial.Ka.Y << ", " << curMesh.MeshMaterial.Ka.Z << "\n";
        file << "Diffuse Color: " << curMesh.MeshMaterial.Kd.X << ", " << curMesh.MeshMaterial.Kd.Y << ", " << curMesh.MeshMaterial.Kd.Z << "\n";
        file << "Specular Color: " << curMesh.MeshMaterial.Ks.X << ", " << curMesh.MeshMaterial.Ks.Y << ", " << curMesh.MeshMaterial.Ks.Z << "\n";
        file << "Specular Exponent: " << curMesh.MeshMaterial.Ns << "\n";
        file << "Optical Density: " << curMesh.MeshMaterial.Ni << "\n";
        file << "Dissolve: " << curMesh.MeshMaterial.d << "\n";
        file << "Illumination: " << curMesh.MeshMaterial.illum << "\n";
        file << "Ambient Texture Map: " << curMesh.MeshMaterial.map_Ka << "\n";
        file << "Diffuse Texture Map: " << curMesh.MeshMaterial.map_Kd << "\n";
        file << "Specular Texture Map: " << curMesh.MeshMaterial.map_Ks << "\n";
        file << "Alpha Texture Map: " << curMesh.MeshMaterial.map_d << "\n";
        file << "Bump Map: " << curMesh.MeshMaterial.map_bump << "\n";
        // Leave a space to separate from the next mesh
        file << "\n";
    }
    // Close File
    file.close();

    return 0;
}