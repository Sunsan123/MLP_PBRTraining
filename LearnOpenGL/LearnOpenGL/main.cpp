#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "shader_s.h"
#include "stb_image.h"
#include "camera.h"
#include "model.h"
#include "mlp_loader.h"

#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace
{
constexpr unsigned int kScreenWidth = 1280;
constexpr unsigned int kScreenHeight = 720;

Camera camera(glm::vec3(0.0f, 0.0f, 4.0f));
float lastX = static_cast<float>(kScreenWidth) / 2.0f;
float lastY = static_cast<float>(kScreenHeight) / 2.0f;
bool firstMouse = true;

float deltaTime = 0.0f;
float lastFrame = 0.0f;

struct PbrTextureSet
{
    unsigned int albedo = 0;
    unsigned int normal = 0;
    unsigned int metallic = 0;
    unsigned int roughness = 0;
    unsigned int ao = 0;
};

struct Light
{
    glm::vec3 position;
    glm::vec3 color;
    float intensity = 1.0f;
};

enum class RenderMode
{
    GroundTruth,
    Mlp
};

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (firstMouse)
    {
        lastX = static_cast<float>(xpos);
        lastY = static_cast<float>(ypos);
        firstMouse = false;
    }

    float xoffset = static_cast<float>(xpos) - lastX;
    float yoffset = lastY - static_cast<float>(ypos);
    lastX = static_cast<float>(xpos);
    lastY = static_cast<float>(ypos);

    camera.ProcessMouseMovement(xoffset, yoffset);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(static_cast<float>(yoffset));
}

void processInput(GLFWwindow* window, RenderMode& mode)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, true);
    }

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    {
        camera.ProcessKeyboard(FORWARD, deltaTime);
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    {
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    {
        camera.ProcessKeyboard(LEFT, deltaTime);
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    {
        camera.ProcessKeyboard(RIGHT, deltaTime);
    }

    if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS)
    {
        mode = RenderMode::GroundTruth;
    }
    if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS)
    {
        mode = RenderMode::Mlp;
    }
}

unsigned int LoadTextureFromFile(const std::string& path)
{
    int width = 0;
    int height = 0;
    int channels = 0;
    stbi_uc* data = stbi_load(path.c_str(), &width, &height, &channels, 0);
    if (!data)
    {
        std::cerr << "Failed to load texture: " << path << std::endl;
        return 0;
    }

    GLenum format = GL_RGB;
    if (channels == 1)
    {
        format = GL_RED;
    }
    else if (channels == 3)
    {
        format = GL_RGB;
    }
    else if (channels == 4)
    {
        format = GL_RGBA;
    }

    unsigned int textureID = 0;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    stbi_image_free(data);
    return textureID;
}

std::optional<PbrTextureSet> LoadDefaultTextures(const std::string& directory)
{
    PbrTextureSet textures;
    textures.albedo = LoadTextureFromFile(directory + "/default_albedo.ppm");
    textures.normal = LoadTextureFromFile(directory + "/default_normal.ppm");
    textures.metallic = LoadTextureFromFile(directory + "/default_metallic.ppm");
    textures.roughness = LoadTextureFromFile(directory + "/default_roughness.ppm");
    textures.ao = LoadTextureFromFile(directory + "/default_ao.ppm");

    if (textures.albedo == 0 || textures.normal == 0 || textures.metallic == 0 ||
        textures.roughness == 0 || textures.ao == 0)
    {
        return std::nullopt;
    }
    return textures;
}

void UploadLights(const Shader& shader, const std::vector<Light>& lights)
{
    shader.use();
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> colors;
    std::vector<float> intensities;
    positions.reserve(lights.size());
    colors.reserve(lights.size());
    intensities.reserve(lights.size());
    for (const auto& light : lights)
    {
        positions.push_back(light.position);
        colors.push_back(light.color);
        intensities.push_back(light.intensity);
    }
    shader.setVec3Array("lightPositions", positions);
    shader.setVec3Array("lightColors", colors);
    shader.setFloatArray("lightIntensities", intensities);
}

void UploadMlpWeights(const Shader& shader, const MlpNetwork& network)
{
    shader.use();
    shader.setFloatArray("uInputMean", network.inputMean);

    std::vector<float> invStd;
    invStd.reserve(network.inputStd.size());
    for (float value : network.inputStd)
    {
        invStd.push_back(value > 1e-6f ? 1.0f / value : 1.0f);
    }
    shader.setFloatArray("uInputInvStd", invStd);
    shader.setFloatArray("uLayer0Weights", network.layer0Weights);
    shader.setFloatArray("uLayer0Bias", network.layer0Bias);
    shader.setFloatArray("uLayer1Weights", network.layer1Weights);
    shader.setFloatArray("uLayer1Bias", network.layer1Bias);
    shader.setFloatArray("uLayer2Weights", network.layer2Weights);
    shader.setFloatArray("uLayer2Bias", network.layer2Bias);
}
}

int main(int argc, char** argv)
{
    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(kScreenWidth, kScreenHeight, "MLP PBR Training", nullptr, nullptr);
    if (window == nullptr)
    {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
    {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    glEnable(GL_DEPTH_TEST);

    stbi_set_flip_vertically_on_load(true);

    Shader groundTruthShader("shaders/pbr_common.vert", "shaders/pbr_ground_truth.frag");
    Shader mlpShader("shaders/pbr_common.vert", "shaders/pbr_mlp.frag");

    groundTruthShader.use();
    groundTruthShader.setInt("albedoMap", 0);
    groundTruthShader.setInt("normalMap", 1);
    groundTruthShader.setInt("metallicMap", 2);
    groundTruthShader.setInt("roughnessMap", 3);
    groundTruthShader.setInt("aoMap", 4);

    mlpShader.use();
    mlpShader.setInt("albedoMap", 0);
    mlpShader.setInt("normalMap", 1);
    mlpShader.setInt("metallicMap", 2);
    mlpShader.setInt("roughnessMap", 3);
    mlpShader.setInt("aoMap", 4);

    std::string modelPath = "assets/models/sphere.obj";
    if (argc > 1)
    {
        modelPath = argv[1];
    }

    Model model;
    std::string modelError;
    if (!model.LoadFromFile(modelPath, modelError))
    {
        std::cerr << modelError << std::endl;
        return -1;
    }

    auto textures = LoadDefaultTextures("assets/textures");
    if (!textures.has_value())
    {
        std::cerr << "Failed to load default PBR textures" << std::endl;
        return -1;
    }

    MlpNetwork network;
    std::string weightError;
    if (!LoadMlpWeights("assets/mlp/mlp_weights.txt", network, weightError))
    {
        std::cerr << weightError << std::endl;
        return -1;
    }

    UploadMlpWeights(mlpShader, network);

    std::vector<Light> lights = {
        { glm::vec3(-10.0f, 10.0f, 10.0f), glm::vec3(300.0f, 300.0f, 300.0f), 1.0f },
        { glm::vec3(10.0f, 10.0f, 10.0f), glm::vec3(300.0f, 300.0f, 300.0f), 1.0f },
        { glm::vec3(-10.0f, -10.0f, 10.0f), glm::vec3(300.0f, 300.0f, 300.0f), 1.0f },
        { glm::vec3(10.0f, -10.0f, 10.0f), glm::vec3(300.0f, 300.0f, 300.0f), 1.0f }
    };

    UploadLights(groundTruthShader, lights);
    UploadLights(mlpShader, lights);

    RenderMode mode = RenderMode::GroundTruth;
    std::cout << "Controls:\n"
              << "  W/A/S/D : move camera\n"
              << "  Mouse   : look around\n"
              << "  Scroll  : adjust FOV\n"
              << "  1       : Ground Truth shader\n"
              << "  2       : MLP shader\n";

    while (!glfwWindowShouldClose(window))
    {
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        processInput(window, mode);

        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom),
                                                static_cast<float>(kScreenWidth) / static_cast<float>(kScreenHeight),
                                                0.1f, 100.0f);
        glm::mat4 view = camera.GetViewMatrix();
        glm::mat4 modelMatrix = glm::mat4(1.0f);

        Shader* activeShader = (mode == RenderMode::GroundTruth) ? &groundTruthShader : &mlpShader;
        activeShader->use();
        activeShader->setMat4("projection", projection);
        activeShader->setMat4("view", view);
        activeShader->setMat4("model", modelMatrix);
        activeShader->setVec3("camPos", camera.Position);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textures->albedo);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, textures->normal);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, textures->metallic);
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, textures->roughness);
        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_2D, textures->ao);

        model.Draw();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
