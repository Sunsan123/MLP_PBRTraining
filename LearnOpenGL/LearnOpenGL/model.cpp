#include "model.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <cmath>

namespace
{
struct VertexIndex
{
    int position = 0;
    int texcoord = 0;
    int normal = 0;
};

VertexIndex ParseIndex(const std::string& token)
{
    VertexIndex index;
    std::stringstream ss(token);
    std::string value;
    if (std::getline(ss, value, '/'))
    {
        index.position = std::stoi(value);
    }
    if (std::getline(ss, value, '/'))
    {
        if (!value.empty())
        {
            index.texcoord = std::stoi(value);
        }
    }
    if (std::getline(ss, value, '/'))
    {
        if (!value.empty())
        {
            index.normal = std::stoi(value);
        }
    }
    return index;
}

unsigned int ToPositiveIndex(int index, std::size_t count)
{
    if (index > 0)
    {
        return static_cast<unsigned int>(index - 1);
    }
    if (index < 0)
    {
        return static_cast<unsigned int>(count + index);
    }
    return 0;
}
}

Mesh::Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices)
    : m_vertices(std::move(vertices)), m_indices(std::move(indices))
{
    setupMesh();
}

Mesh::Mesh(Mesh&& other) noexcept
    : m_vertices(std::move(other.m_vertices)),
      m_indices(std::move(other.m_indices)),
      m_VAO(other.m_VAO),
      m_VBO(other.m_VBO),
      m_EBO(other.m_EBO)
{
    other.m_VAO = 0;
    other.m_VBO = 0;
    other.m_EBO = 0;
}

Mesh& Mesh::operator=(Mesh&& other) noexcept
{
    if (this != &other)
    {
        glDeleteVertexArrays(1, &m_VAO);
        glDeleteBuffers(1, &m_VBO);
        glDeleteBuffers(1, &m_EBO);

        m_vertices = std::move(other.m_vertices);
        m_indices = std::move(other.m_indices);
        m_VAO = other.m_VAO;
        m_VBO = other.m_VBO;
        m_EBO = other.m_EBO;

        other.m_VAO = 0;
        other.m_VBO = 0;
        other.m_EBO = 0;
    }
    return *this;
}

Mesh::~Mesh()
{
    if (m_VAO != 0)
    {
        glDeleteVertexArrays(1, &m_VAO);
    }
    if (m_VBO != 0)
    {
        glDeleteBuffers(1, &m_VBO);
    }
    if (m_EBO != 0)
    {
        glDeleteBuffers(1, &m_EBO);
    }
}

void Mesh::setupMesh()
{
    glGenVertexArrays(1, &m_VAO);
    glGenBuffers(1, &m_VBO);
    glGenBuffers(1, &m_EBO);

    glBindVertexArray(m_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, m_vertices.size() * sizeof(Vertex), m_vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_indices.size() * sizeof(unsigned int), m_indices.data(), GL_STATIC_DRAW);

    const std::size_t stride = sizeof(Vertex);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(offsetof(Vertex, Position)));

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(offsetof(Vertex, Normal)));

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(offsetof(Vertex, TexCoords)));

    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(offsetof(Vertex, Tangent)));

    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(offsetof(Vertex, Bitangent)));

    glBindVertexArray(0);
}

void Mesh::Draw() const
{
    glBindVertexArray(m_VAO);
    glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(m_indices.size()), GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);
}

bool Model::LoadFromFile(const std::string& path, std::string& errorMessage)
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        errorMessage = "Failed to open OBJ file: " + path;
        return false;
    }

    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texcoords;
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    std::unordered_map<std::string, unsigned int> uniqueVertexMap;

    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty() || line[0] == '#')
        {
            continue;
        }
        std::stringstream ss(line);
        std::string prefix;
        ss >> prefix;
        if (prefix == "v")
        {
            glm::vec3 position;
            ss >> position.x >> position.y >> position.z;
            positions.push_back(position);
        }
        else if (prefix == "vn")
        {
            glm::vec3 normal;
            ss >> normal.x >> normal.y >> normal.z;
            normals.push_back(normal);
        }
        else if (prefix == "vt")
        {
            glm::vec2 tex;
            ss >> tex.x >> tex.y;
            texcoords.push_back(tex);
        }
        else if (prefix == "f")
        {
            std::vector<std::string> tokens;
            std::string token;
            while (ss >> token)
            {
                tokens.push_back(token);
            }
            if (tokens.size() < 3)
            {
                continue;
            }
            // Triangulate polygon faces if needed
            for (std::size_t i = 1; i + 1 < tokens.size(); ++i)
            {
                const std::string& a = tokens[0];
                const std::string& b = tokens[i];
                const std::string& c = tokens[i + 1];
                const std::string* faceTokens[3] = { &a, &b, &c };
                for (int k = 0; k < 3; ++k)
                {
                    auto it = uniqueVertexMap.find(*faceTokens[k]);
                    if (it == uniqueVertexMap.end())
                    {
                        VertexIndex idx = ParseIndex(*faceTokens[k]);
                        Vertex vertex{};
                        vertex.Position = positions[ToPositiveIndex(idx.position, positions.size())];
                        if (!texcoords.empty() && idx.texcoord != 0)
                        {
                            vertex.TexCoords = texcoords[ToPositiveIndex(idx.texcoord, texcoords.size())];
                        }
                        else
                        {
                            vertex.TexCoords = glm::vec2(0.0f);
                        }
                        if (!normals.empty() && idx.normal != 0)
                        {
                            vertex.Normal = normals[ToPositiveIndex(idx.normal, normals.size())];
                        }
                        else
                        {
                            vertex.Normal = glm::vec3(0.0f, 0.0f, 1.0f);
                        }
                        vertex.Tangent = glm::vec3(0.0f);
                        vertex.Bitangent = glm::vec3(0.0f);
                        unsigned int newIndex = static_cast<unsigned int>(vertices.size());
                        uniqueVertexMap.emplace(*faceTokens[k], newIndex);
                        vertices.push_back(vertex);
                        indices.push_back(newIndex);
                    }
                    else
                    {
                        indices.push_back(it->second);
                    }
                }
            }
        }
    }

    if (vertices.empty() || indices.empty())
    {
        errorMessage = "OBJ file contained no geometry: " + path;
        return false;
    }

    std::vector<glm::vec3> tanAccum(vertices.size(), glm::vec3(0.0f));
    std::vector<glm::vec3> bitanAccum(vertices.size(), glm::vec3(0.0f));

    for (std::size_t i = 0; i + 2 < indices.size(); i += 3)
    {
        Vertex& v0 = vertices[indices[i]];
        Vertex& v1 = vertices[indices[i + 1]];
        Vertex& v2 = vertices[indices[i + 2]];

        glm::vec3 edge1 = v1.Position - v0.Position;
        glm::vec3 edge2 = v2.Position - v0.Position;
        glm::vec2 deltaUV1 = v1.TexCoords - v0.TexCoords;
        glm::vec2 deltaUV2 = v2.TexCoords - v0.TexCoords;

        float determinant = deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x;
        if (std::abs(determinant) < 1e-8f)
        {
            continue;
        }
        float invDet = 1.0f / determinant;
        glm::vec3 tangent = invDet * (deltaUV2.y * edge1 - deltaUV1.y * edge2);
        glm::vec3 bitangent = invDet * (-deltaUV2.x * edge1 + deltaUV1.x * edge2);

        tanAccum[indices[i]] += tangent;
        tanAccum[indices[i + 1]] += tangent;
        tanAccum[indices[i + 2]] += tangent;

        bitanAccum[indices[i]] += bitangent;
        bitanAccum[indices[i + 1]] += bitangent;
        bitanAccum[indices[i + 2]] += bitangent;
    }

    for (std::size_t i = 0; i < vertices.size(); ++i)
    {
        vertices[i].Tangent = glm::normalize(tanAccum[i]);
        vertices[i].Bitangent = glm::normalize(bitanAccum[i]);
        if (glm::length(vertices[i].Tangent) < 1e-4f)
        {
            vertices[i].Tangent = glm::vec3(1.0f, 0.0f, 0.0f);
        }
        if (glm::length(vertices[i].Bitangent) < 1e-4f)
        {
            vertices[i].Bitangent = glm::vec3(0.0f, 1.0f, 0.0f);
        }
    }

    m_meshes.clear();
    m_meshes.emplace_back(std::move(vertices), std::move(indices));
    m_sourcePath = path;
    return true;
}

void Model::Draw() const
{
    for (const auto& mesh : m_meshes)
    {
        mesh.Draw();
    }
}
