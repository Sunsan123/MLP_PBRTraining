# MLP_PBRTraining

利用多层感知机（MLP）替换 UE4 风格的 PBR BRDF 函数，在 OpenGL 中实现可切换的 “Ground Truth” 与 “MLP” 渲染模式。项目涵盖数据采集、离线训练以及实时推理三大环节。

## 工程结构

```
LearnOpenGL/
 └─ LearnOpenGL/
     ├─ assets/                 # 默认模型、贴图、MLP 权重
     │   ├─ mlp/
     │   ├─ models/
     │   └─ textures/
     ├─ shaders/                # Ground Truth / MLP GLSL 着色器
     ├─ main.cpp                # OpenGL 应用入口
     ├─ model.*                 # 简易 OBJ 加载与切线计算
     └─ mlp_loader.*            # 文本权重加载工具
tools/
 ├─ train_mlp_brdf.py          # 基于 numpy 的高性能训练脚本（需额外依赖）
 ├─ train_mlp_brdf_pure.py     # 纯标准库实现的训练脚本（默认可运行）
 └─ convert_weights_to_txt.py  # JSON → 文本权重转换脚本
```

## 数据生成与离线训练

1. **采样数据**：`tools/train_mlp_brdf_pure.py` / `train_mlp_brdf.py` 会随机生成粗糙度、金属度、法线与视角方向，调用与着色器一致的 UE4 PBR 模型输出 `RGB` 辐射度，构建输入特征 → 输出颜色的数据集。
2. **训练网络**：默认脚本使用三层（输入 8 → 32 → 32 → 输出 3）MLP，并输出均值、标准差与每层权重、偏置。
3. **导出权重**：权重以 `assets/mlp/mlp_weights.json` 保存，同时利用 `convert_weights_to_txt.py` 生成渲染阶段读取的 `mlp_weights.txt` 文本文件。

> **性能提示**：若环境具备 `numpy`，推荐运行 `train_mlp_brdf.py`；无法安装第三方库时，可使用纯 Python 版本 `train_mlp_brdf_pure.py`（默认已执行并生成示例权重）。

## 实时推理应用

`LearnOpenGL/LearnOpenGL/main.cpp` 构建了一个 OpenGL 3.3 Core Profile 程序，可加载 OBJ 模型及五张 PBR 贴图，并提供两种渲染管线：

- **Ground Truth**：`shaders/pbr_ground_truth.frag` 实现 UE4 风格的 Cook-Torrance BRDF 与 GGX/SCHLICK 几何函数；
- **MLP**：`shaders/pbr_mlp.frag` 将 BRDF 核心替换为 MLP 前向推理，所有权重由 CPU 侧加载成 uniform 数组。

### 运行方式

1. 进入 `LearnOpenGL/LearnOpenGL` 可执行目录，准备自定义模型或使用默认 `assets/models/sphere.obj`；
2. 将 PBR 贴图（反照率、法线、金属度、粗糙度、AO）放置于 `assets/textures`，如需替换请覆盖默认 `ppm` 文件；
3. 确认 `assets/mlp/mlp_weights.txt` 已就绪（可通过训练脚本重新生成）；
4. 使用 Visual Studio 或命令行编译并运行项目。

应用内的默认按键：

- `W/A/S/D`：自由移动；
- 鼠标移动 / 滚轮：视角与 FOV；
- `1`：切换至 Ground Truth 着色；
- `2`：切换至 MLP 着色。

## 重新训练并部署

```bash
# 纯标准库训练（默认环境即可运行）
python tools/train_mlp_brdf_pure.py

# 若已安装 numpy，可使用更快的实现
python tools/train_mlp_brdf.py --samples 200000 --epochs 200

# 将 JSON 权重转换为渲染读取的文本格式
python tools/convert_weights_to_txt.py
```

训练完成后，将生成的新 `mlp_weights.txt` 复制/覆盖至 `assets/mlp/`，重新启动 OpenGL 应用即可使用最新网络进行实时推理。
