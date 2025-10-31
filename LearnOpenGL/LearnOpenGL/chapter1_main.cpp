#include <glad/glad.h> 
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "shader_s.h"
#include "stb_image.h"
#include "camera.h"

#include <iostream> // ���� C++ ��׼��������������⣬�����ڿ���̨�����Ϣ��

#if USE_CHAPTER1 1

// ����ԭ������
// ----------------
// �����ڴ�С�ı�ʱ���˻ص������ᱻ���á�
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
// �������������¼��ĺ�����
void processInput(GLFWwindow* window);

void mouse_callback(GLFWwindow* window, double xpos, double ypos);

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

// ����
// ������Ļ�����ڣ��Ŀ�߶ȡ�
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

//timing
float deltaTime = 0.0f; // ��ǰ֡����һ֡��ʱ���
float lastFrame = 0.0f; // ��һ֡��ʱ��

// camera
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

//vertex
float vertices[] = {
    -1.0f,  1.0f, -1.0f,   0.0f, 1.0f,
     1.0f,  1.0f,  1.0f,   1.0f, 0.0f,
     1.0f,  1.0f, -1.0f,   1.0f, 1.0f,
     1.0f,  1.0f,  1.0f,   1.0f, 1.0f,
    -1.0f, -1.0f,  1.0f,   0.0f, 0.0f,
     1.0f, -1.0f,  1.0f,   1.0f, 0.0f,
    -1.0f,  1.0f,  1.0f,   0.0f, 1.0f,
    -1.0f, -1.0f, -1.0f,   1.0f, 0.0f,
     1.0f, -1.0f, -1.0f,   1.0f, 1.0f,
    -1.0f, -1.0f, -1.0f,   0.0f, 1.0f,
     1.0f, -1.0f,  1.0f,   0.0f, 0.0f,
     1.0f, -1.0f, -1.0f,   1.0f, 0.0f,
    -1.0f, -1.0f, -1.0f,   0.0f, 0.0f,
    -1.0f,  1.0f,  1.0f,   0.0f, 0.0f,
    -1.0f,  1.0f, -1.0f,   1.0f, 1.0f,
     1.0f,  1.0f,  1.0f,   0.0f, 1.0f
};


unsigned int indices[] = {
     0,  1,  2,
     3,  4,  5,
     6,  7,  4,
     8,  4,  9,
     2, 10, 11,
     0, 11, 12,
     0, 13,  1,
     3,  6,  4,
     6, 14,  7,
     8,  5,  4,
     2, 15, 10,
     0,  2, 11
};

glm::vec3 cubePositions[] = {
  glm::vec3(0.0f,  0.0f,  0.0f),
  glm::vec3(2.0f,  5.0f, -15.0f),
  glm::vec3(-1.5f, -2.2f, -2.5f),
  glm::vec3(-3.8f, -2.0f, -12.3f),
  glm::vec3(2.4f, -0.4f, -3.5f),
  glm::vec3(-1.7f,  3.0f, -7.5f),
  glm::vec3(1.3f, -2.0f, -2.5f),
  glm::vec3(1.5f,  2.0f, -2.5f),
  glm::vec3(1.5f,  0.2f, -1.5f),
  glm::vec3(-1.3f,  1.0f, -1.5f)
};

//--------------------------��-----��-----��---------------------------------//
int main()
{
    // glfw: ��ʼ��������
    // ------------------------------
    glfwInit(); // ��ʼ�� GLFW �⡣
    // ���� GLFW �Ĵ�����ʾ��Hint����ָ��������Ҫʹ�õ� OpenGL �汾��
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    // ���� GLFW ����Ҫʹ�ú���ģʽ��Core-profile��������ζ������ֻ��ʹ���ִ��� OpenGL ���ܡ�
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);


    // glfw: ��������
    // --------------------
    // ����һ�����ڶ��󡣲����ֱ��ǣ����ߡ����ڱ��⣬�����������Ļ�͹��������ģ��˴���Ϊ NULL��
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == NULL) // ��鴰���Ƿ�ɹ�������
    {
        std::cout << "Failed to create GLFW window" << std::endl; // ���ʧ�ܣ����������Ϣ������̨��
        glfwTerminate(); // ��ֹ GLFW��
        return -1; // ���� -1 ��ʾ�������
    }
    // �����Ǵ����Ĵ��ڵ�����������Ϊ��ǰ�̵߳��������ġ�
    glfwMakeContextCurrent(window);
    // ע�� framebuffer_size_callback ������ÿ�����ڴ�С�ı�ʱ��GLFW �ͻ�������������
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);


    // glad: �������� OpenGL ����ָ��
    // --------------------------------------- 
    // ��ʼ�� GLAD�����Ǵ��ݸ� GLAD �������� OpenGL ������ַ�� GLFW ������
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl; // �����ʼ��ʧ�ܣ����������Ϣ��
        return -1; // ���� -1 ��ʾ�������
    }

    //--------------����ȫ��״̬------------------//
     glEnable(GL_DEPTH_TEST);

     glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
     //--------------������ɫ��------------------//
    Shader ourShader("3.3.shader.vs", "3.3.shader.fs"); // you can name your shader files however you like

     //-------------����Buffer------------------//
    unsigned int VBO, VAO, EBO;
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    glGenVertexArrays(1, &VAO);


    // 1. ��VAO
    glBindVertexArray(VAO);
    //2. ��VBO����������
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    // 3. ��EBO����������
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
    // 4. �趨��������ָ��
    // λ������
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    //uv
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    //���
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    //----------------����-----------------//
    unsigned int texture1,texture2;
    glGenTextures(1, &texture1);
    glBindTexture(GL_TEXTURE_2D, texture1);
    // Ϊ��ǰ�󶨵�����������û��ơ����˷�ʽ
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // ���ز���������
    int width, height, nrChannels;
    stbi_set_flip_vertically_on_load(true);
    unsigned char* data = stbi_load("container.jpg", &width, &height, &nrChannels, 0);
    if (data)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else
    {
        std::cout << "Failed to load texture1" << std::endl;
    }
    stbi_image_free(data);
    //--//
    glGenTextures(1, &texture2);
    glBindTexture(GL_TEXTURE_2D, texture2);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    data = stbi_load("awesomeface.png", &width, &height, &nrChannels, 0);
    if (data)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else
    {
        std::cout << "Failed to load texture2" << std::endl;
    }
    stbi_image_free(data);

    ourShader.use(); // ��Ҫ����������uniform����֮ǰ������ɫ������
    ourShader.setInt("texture1", 0);
    ourShader.setInt("texture2", 1); 

    //----------------------����任-----------------------//
    glm::mat4 model;
    model = glm::rotate(model, glm::radians(-55.0f), glm::vec3(1.0f, 0.0f, 0.0f));

    glm::mat4 view;

    glm::mat4 projection;


    //--------------------------��-----Ⱦ-----ѭ-----��---------------------------------//
    // ѭ����һֱִ�У�ֱ�����Ǹ��� GLFW ����Ӧ�ùرա�
    while (!glfwWindowShouldClose(window))
    {
        //--------- deltaTime----------//
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        //--------- ���봦��----------//
        processInput(window); // ��ÿһ֡��ѭ����ÿһ�ε������м�����롣
        glfwSetCursorPosCallback(window, mouse_callback);
        glfwSetScrollCallback(window, scroll_callback);
        //--------- ÿ֡Buffer����----------//
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        //--------- ÿ֡��������----------//
       //
        view = camera.GetViewMatrix();
        projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);

        // --------- ��Ⱦָ��----------//
        ourShader.use();

        // ����uniform��ɫ
        float timeValue = glfwGetTime();
        float greenValue = sin(timeValue) / 2.0f + 0.5f;
        ourShader.setVec4("ourShader",0.0f,greenValue, 0.0f, 1.0f);
        ourShader.setMat4("model", model);
        ourShader.setMat4("view", view);
        ourShader.setMat4("projection",  projection);
        //--------------��---------------//
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture1);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, texture2);

        //--------------��Ⱦ---------------//
        glBindVertexArray(VAO);
        for (unsigned int i = 0; i < 10; i++)
        {
            glm::mat4 model;
            model = glm::translate(model, cubePositions[i]);
            float angle = 20.0f * i;
            model = glm::rotate(model, glm::radians(angle), glm::vec3(1.0f, 0.3f, 0.5f));
            ourShader.setMat4("model",model);

            glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
        }
        //--------------���---------------//
        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        // glfw: ��������������ѯ IO �¼�����������/�ͷš�����ƶ��ȣ�
        // -------------------------------------------------------------------------------
        // ������ɫ��������һ���洢��ÿ��������ɫֵ�Ĵ󻺳�������
        // ˫������ϵͳ��ǰ��������ʾͼ�񣬺󻺳������л��ơ����������Ա��⻭��˺�ѡ�
        glfwSwapBuffers(window);
        // ����Ƿ��д����κ��¼�����������롢����ƶ��ȣ��������ö�Ӧ�Ļص�������
        glfwPollEvents();
    }

    // glfw: ��ֹ�����������ǰ����� GLFW ��Դ��
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0; // ��������������
}

// �����������룺��ѯ GLFW ����һ֡���Ƿ�����صİ���������/�ͷţ���������Ӧ�ķ�Ӧ��
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}


// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(static_cast<float>(yoffset));
}

#endif