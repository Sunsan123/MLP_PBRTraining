#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <string>
#include <vector>

struct Vertex
{
    glm::vec3 Position;
    glm::vec3 Normal;
    glm::vec2 TexCoords;
    glm::vec3 Tangent;
    glm::vec3 Bitangent;
};

class Mesh
{
public:
    Mesh() = default;
    Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices);
    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;
    Mesh(Mesh&& other) noexcept;
    Mesh& operator=(Mesh&& other) noexcept;
    ~Mesh();

    void Draw() const;

private:
    void setupMesh();

    std::vector<Vertex> m_vertices;
    std::vector<unsigned int> m_indices;
    unsigned int m_VAO = 0;
    unsigned int m_VBO = 0;
    unsigned int m_EBO = 0;
};

class Model
{
public:
    Model() = default;
    bool LoadFromFile(const std::string& path, std::string& errorMessage);
    void Draw() const;
    const std::string& GetSourcePath() const { return m_sourcePath; }

private:
    std::vector<Mesh> m_meshes;
    std::string m_sourcePath;
};
