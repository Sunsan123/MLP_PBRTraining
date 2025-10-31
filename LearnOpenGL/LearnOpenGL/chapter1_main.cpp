#include <glad/glad.h> 
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "shader_s.h"
#include "stb_image.h"
#include "camera.h"

#include <iostream> // 引入 C++ 标准输入输出流函数库，用于在控制台输出信息。

#if USE_CHAPTER1 1

// 函数原型声明
// ----------------
// 当窗口大小改变时，此回调函数会被调用。
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
// 处理所有输入事件的函数。
void processInput(GLFWwindow* window);

void mouse_callback(GLFWwindow* window, double xpos, double ypos);

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

// 设置
// 定义屏幕（窗口）的宽高度。
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

//timing
float deltaTime = 0.0f; // 当前帧与上一帧的时间差
float lastFrame = 0.0f; // 上一帧的时间

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

//--------------------------主-----函-----数---------------------------------//
int main()
{
    // glfw: 初始化与设置
    // ------------------------------
    glfwInit(); // 初始化 GLFW 库。
    // 设置 GLFW 的窗口提示（Hint），指定我们想要使用的 OpenGL 版本。
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    // 告诉 GLFW 我们要使用核心模式（Core-profile），这意味着我们只能使用现代的 OpenGL 功能。
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);


    // glfw: 创建窗口
    // --------------------
    // 创建一个窗口对象。参数分别是：宽、高、窗口标题，最后两个是屏幕和共享上下文，此处设为 NULL。
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == NULL) // 检查窗口是否成功创建。
    {
        std::cout << "Failed to create GLFW window" << std::endl; // 如果失败，输出错误信息到控制台。
        glfwTerminate(); // 终止 GLFW。
        return -1; // 返回 -1 表示程序出错。
    }
    // 将我们创建的窗口的上下文设置为当前线程的主上下文。
    glfwMakeContextCurrent(window);
    // 注册 framebuffer_size_callback 函数。每当窗口大小改变时，GLFW 就会调用这个函数。
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);


    // glad: 加载所有 OpenGL 函数指针
    // --------------------------------------- 
    // 初始化 GLAD。我们传递给 GLAD 用来加载 OpenGL 函数地址的 GLFW 函数。
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl; // 如果初始化失败，输出错误信息。
        return -1; // 返回 -1 表示程序出错。
    }

    //--------------配置全局状态------------------//
     glEnable(GL_DEPTH_TEST);

     glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
     //--------------编译着色器------------------//
    Shader ourShader("3.3.shader.vs", "3.3.shader.fs"); // you can name your shader files however you like

     //-------------顶点Buffer------------------//
    unsigned int VBO, VAO, EBO;
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    glGenVertexArrays(1, &VAO);


    // 1. 绑定VAO
    glBindVertexArray(VAO);
    //2. 绑定VBO并发送数据
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    // 3. 绑定EBO并发送数据
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
    // 4. 设定顶点属性指针
    // 位置属性
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    //uv
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    //解绑
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    //----------------纹理-----------------//
    unsigned int texture1,texture2;
    glGenTextures(1, &texture1);
    glBindTexture(GL_TEXTURE_2D, texture1);
    // 为当前绑定的纹理对象设置环绕、过滤方式
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // 加载并生成纹理
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

    ourShader.use(); // 不要忘记在设置uniform变量之前激活着色器程序！
    ourShader.setInt("texture1", 0);
    ourShader.setInt("texture2", 1); 

    //----------------------矩阵变换-----------------------//
    glm::mat4 model;
    model = glm::rotate(model, glm::radians(-55.0f), glm::vec3(1.0f, 0.0f, 0.0f));

    glm::mat4 view;

    glm::mat4 projection;


    //--------------------------渲-----染-----循-----环---------------------------------//
    // 循环会一直执行，直到我们告诉 GLFW 窗口应该关闭。
    while (!glfwWindowShouldClose(window))
    {
        //--------- deltaTime----------//
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        //--------- 输入处理----------//
        processInput(window); // 在每一帧（循环的每一次迭代）中检查输入。
        glfwSetCursorPosCallback(window, mouse_callback);
        glfwSetScrollCallback(window, scroll_callback);
        //--------- 每帧Buffer处理----------//
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        //--------- 每帧更新数据----------//
       //
        view = camera.GetViewMatrix();
        projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);

        // --------- 渲染指令----------//
        ourShader.use();

        // 更新uniform颜色
        float timeValue = glfwGetTime();
        float greenValue = sin(timeValue) / 2.0f + 0.5f;
        ourShader.setVec4("ourShader",0.0f,greenValue, 0.0f, 1.0f);
        ourShader.setMat4("model", model);
        ourShader.setMat4("view", view);
        ourShader.setMat4("projection",  projection);
        //--------------绑定---------------//
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture1);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, texture2);

        //--------------渲染---------------//
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
        //--------------解绑---------------//
        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        // glfw: 交换缓冲区并轮询 IO 事件（按键按下/释放、鼠标移动等）
        // -------------------------------------------------------------------------------
        // 交换颜色缓冲区（一个存储着每个像素颜色值的大缓冲区）。
        // 双缓冲区系统：前缓冲区显示图像，后缓冲区进行绘制。交换它们以避免画面撕裂。
        glfwSwapBuffers(window);
        // 检查是否有触发任何事件（如键盘输入、鼠标移动等），并调用对应的回调函数。
        glfwPollEvents();
    }

    // glfw: 终止，清除所有先前分配的 GLFW 资源。
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0; // 程序正常结束。
}

// 处理所有输入：查询 GLFW 在这一帧中是否有相关的按键被按下/释放，并做出相应的反应。
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