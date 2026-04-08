# ChromaManga MCP Server

<div align="center">

**通过 Model Context Protocol 提供 AI 漫画自动上色服务**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-1.0-blue.svg)](https://modelcontextprotocol.io/)

简体中文

[功能特性](#功能特性) • [快速开始](#快速开始) • [MCP 工具](#mcp-工具) • [使用示例](#使用示例) • [技术架构](#技术架构)

</div>

---

## 项目简介

本项目基于 [ChromaManga](https://github.com/TheresaQAQ/ChromaManga) 开发，将其强大的漫画上色能力封装为 **MCP (Model Context Protocol) 服务**，让 AI 助手（如 Claude、Kiro）能够通过自然语言对话完成漫画上色任务。

### 什么是 MCP？

MCP 是一个开放协议，允许 AI 应用与外部工具和数据源无缝集成。通过 MCP，AI 助手可以：
- 调用本地工具和服务
- 访问文件系统和数据库
- 执行复杂的多步骤任务

---

## 功能特性

### 核心上色能力
🎨 **区域提示词控制** - 为背景和每个角色使用不同的提示词，单次推理完成精准上色

🤖 **自动角色识别** - 基于 CLIP 的 ReID 自动识别角色并应用匹配的 LoRA 风格

🎯 **ControlNet 线稿引导** - 高保真度保留原始漫画线稿结构

💬 **对话框文字保护** - 自动检测并保留对话框中的文字

⚡ **Real-ESRGAN 超分辨率** - 可选的 4x 超分辨率输出高质量结果

### MCP 服务特性
🔧 **16 个工具函数** - 涵盖从任务创建到后处理的完整流程

📊 **任务状态管理** - 实时查询任务进度和中间结果

🔄 **流程回退** - 支持回退到任意步骤重新执行

⚙️ **动态参数调整** - 运行时修改提示词和推理参数

🤝 **AI 对话集成** - 在 Kiro、Claude Desktop 等客户端中通过自然语言使用

---

## 快速开始

### 环境要求
- Python 3.10+
- 支持 CUDA 的 GPU，显存 12GB+（推荐）
- 20GB+ 磁盘空间用于模型存储

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/YourUsername/ChromaManga-MCP.git
cd ChromaManga-MCP

# 2. 安装核心依赖
pip install -r requirements.txt

# 3. 安装 MCP 依赖
pip install -r mcp_server/mcp_requirements.txt

# 4. 模型会在首次运行时自动下载
# 或手动下载到 data/models/ 目录：
# - 基础模型：Illustrious-XL
# - ControlNet：xinsir/controlnet-union-sdxl-1.0
# - YOLO 模型：动漫人物检测 + 气泡检测
```

### 配置 MCP 客户端

#### 在 Kiro 中使用

编辑 `~/.kiro/settings/mcp.json`：

```json
{
  "mcpServers": {
    "chromamanga": {
      "command": "python",
      "args": ["path/to/ChromaManga-MCP/mcp_server/chromamanga_mcp_server.py"],
      "env": {
        "PYTHONPATH": "path/to/ChromaManga-MCP"
      },
      "disabled": false,
      "autoApprove": ["get_task_status", "get_config", "analyze_image"]
    }
  }
}
```

#### 在 Claude Desktop 中使用

编辑 Claude Desktop 配置文件：

```json
{
  "mcpServers": {
    "chromamanga": {
      "command": "python",
      "args": ["path/to/ChromaManga-MCP/mcp_server/chromamanga_mcp_server.py"],
      "env": {
        "PYTHONPATH": "path/to/ChromaManga-MCP"
      }
    }
  }
}
```

### 验证安装

```bash
# 运行集成测试
python tests/verify_integration.py

# 测试完整流程（需要 15-20 分钟）
python tests/test_full_pipeline.py data/inputs/manga.png
```

---

## MCP 工具

本服务提供 16 个工具函数，涵盖完整的漫画上色流程：

### 任务管理（3 个）
- `create_task` - 创建新的上色任务
- `get_task_status` - 查询任务状态和进度
- `get_task_result` - 获取最终结果和中间产物

### 流程步骤（7 个）
- `extract_lineart` - 提取线稿用于 ControlNet 引导
- `detect_persons` - 检测图中的人物位置
- `identify_characters` - 使用 CLIP ReID 识别角色
- `detect_bubbles` - 检测对话框保护文字区域
- `generate_masks` - 生成区域蒙版
- `run_inference` - 执行 SDXL + ControlNet 推理（10-20 分钟）
- `postprocess` - 后处理：叠加线稿、恢复文字、超分辨率

### 配置管理（3 个）
- `get_config` - 获取当前配置
- `update_inference_params` - 调整推理参数（步数、CFG、种子等）
- `update_prompt` - 修改角色或背景提示词

### 工具函数（3 个）
- `analyze_image` - 分析图像质量指标
- `reset_to_step` - 回退到指定步骤重新执行

---

## 使用示例

### 示例 1：完整上色流程

在 AI 助手中输入：

```
帮我给这张漫画上色：/path/to/manga.png

请按照完整流程执行：
1. 创建任务
2. 提取线稿
3. 检测人物
4. 识别角色
5. 检测对话框
6. 生成区域蒙版
7. 执行推理
8. 应用后处理
```

AI 会自动调用所有工具完成上色，并告诉你结果保存位置。

### 示例 2：调整颜色

```
User: Sagiri 的头发太暗了，调亮一点

AI 操作：
1. get_config - 查看当前配置
2. update_prompt - 修改 Sagiri 的提示词，添加 "bright, light, vibrant"
3. reset_to_step - 回退到推理步骤
4. run_inference - 重新生成
5. postprocess - 应用后处理
```

### 示例 3：调整参数

```
User: 颜色太淡了，调鲜艳一点

AI 操作：
1. analyze_image - 分析当前结果
2. update_inference_params - 提高 guidance_scale (6.0 → 7.0)
3. reset_to_step - 回退到推理
4. run_inference - 重新生成
```

### 示例 4：查看进度

```
User: 查看任务 abc-123 的状态

AI 调用 get_task_status 显示：
- 当前状态：processing
- 进度：75% (6/8 步骤)
- 当前步骤：run_inference
- 已完成：lineart_extraction, detect_persons, ...
```

---

## 技术架构

### 工作流程

```
┌─────────────────────────────────────────────────────────────────┐
│ MCP 客户端（Kiro / Claude Desktop）                              │
│   • 用户通过自然语言对话                                          │
│   • AI 助手理解意图并调用 MCP 工具                                │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ MCP Server (chromamanga_mcp_server.py)                          │
│   • 16 个工具函数                                                │
│   • 任务状态管理                                                 │
│   • 参数动态调整                                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ ChromaManga 核心引擎                                             │
│                                                                 │
│  1. 输入处理                                                     │
│     • 加载黑白漫画图像                                           │
│     • 提取线稿（ControlNet 兼容格式）                            │
│     • 检测对话框 → 提取文字 mask                                 │
│                                                                 │
│  2. 角色检测与识别                                               │
│     • YOLO 人物检测 → 边界框                                     │
│     • CLIP ReID → 匹配角色模板                                   │
│     • 分配区域专属提示词                                         │
│                                                                 │
│  3. 区域注意力推理                                               │
│     • 构建空间 mask（背景 + 各人物区域）                         │
│     • 编码所有区域提示词                                         │
│     • 注入 RegionalAttnProcessor 到 UNet                         │
│     • 单次 SDXL + ControlNet 推理                                │
│     • 不同提示词路由到不同区域                                   │
│                                                                 │
│  4. 后处理                                                       │
│     • 叠加原始线稿（保持线条清晰度）                             │
│     • 还原对话框文字（inpaint + 印回）                           │
│     • Real-ESRGAN 4x 超分辨率                                    │
│     • 保存最终结果 + 调试可视化                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 核心技术

- **Model Context Protocol (MCP)** - AI 工具集成协议
- **Stable Diffusion XL** - 高质量图像生成
- **ControlNet（线稿）** - 保留漫画线条结构
- **区域注意力** - 自定义注意力处理器实现空间提示词路由
- **CLIP ReID** - 基于视觉相似度的角色识别
- **YOLO** - 人物和对话框检测
- **Real-ESRGAN** - 动漫优化的超分辨率

---

## 项目结构

```
ChromaManga-MCP/
├── core/                    # 核心算法模块
│   ├── colorize.py          # 主上色逻辑
│   ├── config.py            # 配置管理
│   └── task_manager.py      # 任务管理器
│
├── mcp_server/              # MCP 服务器
│   ├── chromamanga_mcp_server.py  # MCP 主服务
│   ├── mcp_requirements.txt       # MCP 依赖
│   └── mcp_config_example.json    # 配置示例
│
├── utils/                   # 工具函数
│   ├── preprocess.py        # 线稿提取与去噪
│   ├── postprocess.py       # 线稿融合
│   ├── character_reid.py    # CLIP 角色识别
│   └── regional_attention.py # 区域注意力处理器
│
├── tests/                   # 测试文件
│   ├── test_mcp_server.py   # MCP 基础测试
│   ├── test_full_pipeline.py # 完整流程测试
│   └── verify_integration.py # 集成验证
│
├── data/                    # 数据目录
│   ├── inputs/              # 输入图片
│   ├── outputs/             # 输出结果
│   ├── loras/               # LoRA 模型
│   ├── models/              # 预训练模型
│   └── refs/                # 角色参考图（用于 ReID）
│
├── docs/                    # 文档
│   ├── MCP_SERVER_README.md # MCP 详细文档
│   ├── QUICK_START.md       # 快速开始指南
│   └── architecture/        # 架构设计文档
│
└── requirements.txt         # Python 依赖
```

---

## 参数调整指南

### 颜色相关
| 问题 | 解决方案 | 参数调整 |
|------|---------|---------|
| 颜色太淡 | 提高提示词遵循度 | `guidance_scale: 6.0 → 7.0` |
| 颜色过饱和 | 降低提示词遵循度 | `guidance_scale: 6.0 → 5.0` |
| 特定角色颜色不对 | 修改角色提示词 | `update_prompt` 添加颜色关键词 |

### 质量相关
| 问题 | 解决方案 | 参数调整 |
|------|---------|---------|
| 细节不够 | 增加推理步数 | `num_inference_steps: 20 → 25` |
| 线稿不清晰 | 提高 ControlNet 强度 | `controlnet_scale: 1.1 → 1.2` |
| 太死板 | 降低 ControlNet 强度 | `controlnet_scale: 1.1 → 0.9` |
| 线稿太重 | 降低叠加强度 | `blend_alpha: 0.15 → 0.1` |

### 背景相关
| 问题 | 解决方案 | 示例提示词 |
|------|---------|-----------|
| 背景太花 | 简化背景提示词 | `"simple white background, minimal"` |
| 背景太单调 | 丰富背景提示词 | `"detailed indoor scene, furniture, ..."` |

---

## 角色自动识别

在 `data/refs/character_name/` 目录下放置参考图：

```
data/refs/
├── sagiri/
│   ├── ref1.png
│   ├── ref2.jpg
│   └── ref3.png
└── another_character/
    └── ref1.png
```

系统会自动：
1. 从参考图提取 CLIP 特征
2. 构建角色模板（缓存在 `data/models/reid_templates.npy`）
3. 将检测到的人物匹配到模板
4. 应用对应的 LoRA 和提示词

---

## 系统要求

- **GPU**：NVIDIA GPU，显存 12GB+（RTX 3060 Ti 或更好）
- **内存**：16GB+ 系统内存
- **存储**：20GB+ 用于模型和缓存
- **操作系统**：Windows 10/11、Linux、macOS（需 CUDA 支持）

---

## 常见问题

### MCP 相关

**MCP Server 无法启动**
- 检查配置文件路径是否正确
- 运行 `python tests/verify_integration.py` 验证
- 查看客户端日志输出

**工具调用失败**
- 确认任务 ID 正确
- 检查是否完成了前置步骤
- 查看错误信息并根据提示操作

### 性能相关

**显存不足（OOM）**
- 在 `core/config.py` 中降低 `max_infer_resolution`
- 启用 `downscale_before_infer = True`
- 关闭其他占用 GPU 的应用

**推理速度慢**
- 确保 CUDA 可用：`torch.cuda.is_available()`
- 安装 xformers：`pip install xformers`
- 使用更少的推理步数（15-20）

### 质量相关

**上色质量差**
- 调整 `controlnet_scale`（尝试 0.9-1.2）
- 调整 `guidance_scale`（尝试 5.5-7.5）
- 检查 LoRA 是否与基础模型兼容
- 确保线稿提取质量（查看 debug 图像）

---

## 输出结果

每个任务会生成完整的中间结果：

```
data/outputs/colored/debug/{task_id}_colored/
├── 00_input.png              # 原始输入
├── 01_bubble_bbox.png         # 对话框检测
├── 02_lineart.png             # 线稿提取
├── 02_person_detection.png    # 人物检测
├── 03_bboxes_reid.png         # 角色识别
├── 04_region_mask.png         # 区域蒙版
├── 05_colored_raw.png         # 原始上色
├── 06_lineart_blend.png       # 叠加线稿
└── 07_final.png               # 最终结果

data/outputs/colored/
└── {filename}_colored.png     # 主输出文件
```

---

## 开发指南

### 添加新的 MCP 工具

1. 在 `list_tools()` 中添加工具定义
2. 在 `call_tool()` 中添加路由
3. 实现对应的 `handle_xxx()` 函数

### 调试 MCP Server

```bash
# 直接运行 MCP Server
python mcp_server/chromamanga_mcp_server.py

# 查看日志（输出到 stderr）
python mcp_server/chromamanga_mcp_server.py 2> mcp_server.log
```

### 运行测试

```bash
# 基础功能测试
python tests/test_mcp_server.py

# 完整流程测试
python tests/test_full_pipeline.py data/inputs/manga.png

# 集成验证
python tests/verify_integration.py
```

---

## 开源协议

本项目采用 MIT 协议 - 详见 [LICENSE](LICENSE) 文件

---

## 致谢

本项目基于以下开源项目开发：

- [ChromaManga](https://github.com/TheresaQAQ/ChromaManga) - 原始漫画上色项目
- [Model Context Protocol](https://modelcontextprotocol.io/) - MCP 协议规范
- [Stable Diffusion XL](https://github.com/Stability-AI/generative-models) by Stability AI
- [ControlNet](https://github.com/lllyasviel/ControlNet) by Lvmin Zhang
- [Illustrious-XL](https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0) by OnomaAI Research
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) by Xintao Wang
- [CLIP](https://github.com/openai/CLIP) by OpenAI
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

---

## 相关文档

- [MCP Server 详细文档](docs/MCP_SERVER_README.md)
- [快速开始指南](docs/QUICK_START.md)
- [API 设计文档](docs/API_DESIGN.md)
- [架构设计文档](docs/architecture/)

---

## 引用

如果您在研究或项目中使用了本项目，请引用：

```bibtex
@software{chromamanga_mcp_2024,
  title={ChromaManga MCP Server: AI-Powered Manga Colorization via Model Context Protocol},
  author={Your Name},
  year={2024},
  url={https://github.com/YourUsername/ChromaManga-MCP}
}
```

---

<div align="center">

**用 ❤️ 为漫画艺术家和 AI 开发者打造**

[报告问题](https://github.com/YourUsername/ChromaManga-MCP/issues) • [功能建议](https://github.com/YourUsername/ChromaManga-MCP/issues) • [贡献代码](https://github.com/YourUsername/ChromaManga-MCP/pulls)

</div>
