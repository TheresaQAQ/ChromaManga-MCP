# 项目 AI 协作规范

> **深度理解项目请先读 `docs/PROJECT_OVERVIEW.md`**（技术白皮书，含论文级架构说明、创新点、关键代码索引）。
> **可视化图解**：`docs/architecture.drawio`（五层分层架构图）+ `docs/pipeline.drawio`（25 步上色流程图）。
> 本文件（CLAUDE.md）是**协作规范**与**操作手册**。

## 语言与回复结构

- 始终使用**简体中文**回复
- Token 使用优先级：快速理解需求，找到根本原因，直接提出最佳方案，避免不必要的询问和冗余输出；保持每次交互简洁有效
- 运用第一性原理思考，拒绝经验主义和路径盲从；若目标模糊请停下讨论，若目标清晰但路径非最优，直接建议更短、更低成本的办法

**所有回答分两部分：**

1. **直接执行** — 按照当前要求和逻辑，直接给出任务结果
2. **深度交互** — 基于底层逻辑对原始需求进行审慎挑战：质疑动机是否偏离目标（XY 问题）、分析当前路径的弊端、给出更优雅的替代方案

## 执行原则

| 原则 | 要求 |
|---|---|
| **默认直接实现** | 收到任务即提供方案并修改代码，不要先问「是否要做」；例外：用户说「不需要实现 / 只要方案 / 先讨论」 |
| **矛盾时选最佳** | 描述矛盾或模糊时，优先级：①用户明确指令 ②不破坏兼容 ③对齐当前实现 ④通用最佳实践；重大架构/模式变更需用户确认 |
| **减少无效对话** | 不重复确认文档中已有的信息；不问可推断的细节；一次性提供完整方案与涉及文件 |

## 独立验证原则（高优先级）

- **禁止以用户判定替代独立验证**：当用户以截图、口述、演示等方式声称某结果成立时，AI 必须通过阅读源码、运行测试、审查日志等手段**独立确认**，再将其作为后续决策依据
- **验证失败时暂停**：若无法独立验证，应明确告知用户"该结论未经验证"，并说明需要什么条件才能验证，而非默认采信
- **结论须标注验证状态**：引用的关键事实，标注 `[已验证]` / `[待验证]` / `[用户声称]`

## 文档/代码一致性规则（硬性）

**README / 工具清单 / `Task.STEPS_ORDER` 必须与 `mcp_server/chromamanga_mcp_server.py::list_tools()` 保持 100% 一致。** 这是不可跳过的硬性约束。

- **触发条件**：只要一次变更涉及以下任意一项，**同一次提交内**必须同步更新受影响的文档：
  - `list_tools()` 中新增 / 删除 / 重命名工具或参数
  - `core/task_manager.py::Task.STEPS_ORDER` 阶段顺序或命名变化
  - `core/colorize.py::colorize_regional()` 主管线阶段划分调整
  - `core/config.py` 对外暴露的参数键名变化（`get_config` 会回读）
  - LoRA 列表 `lora_configs` 的角色索引含义变化（`update_prompt` 的 `character_index` 注释依赖它；`agent/graph.py::SYSTEM_PROMPT` 里的"LoRA 角色索引"小节也依赖它）

- **需要同步的位置**：`README.md`、`README_CN.md`（Features / MCP Tools / Architecture 段）、`CHANGELOG.md`、本文件的"MCP 工具清单"表、`mcp_server/mcp_config_example.json`（若示例依赖新参数）、`agent/graph.py::SYSTEM_PROMPT`（Agent 系统提示词里硬编码了工具名和阶段顺序）、`agent/mock_tools.py`（mock 工具的签名和返回结构必须与真 MCP 工具对齐，否则切回 `AGENT_MCP_MODE=real` 时 Agent 行为会漂移）。
- **当前已知漂移**：`README_CN.md` 和 `README.md` 宣称 "16 个工具"，实际 `list_tools()` 只有 15 个；任一侧修改时顺手校准另一侧。
- **执行顺序**：**先改代码 → 再改文档 → 再提交**。严禁"下次一起改"、"文档只是给人看的，不同步无所谓"。
- **自检动作**：声明任务完成前，必须主动回答一次"本次变更是否触发同步条件？" 若触发，在回复里列出同步点（工具名 / 文档小节）。

## 项目总览

**ChromaManga-MCP** 是 AI 漫画自动上色服务，Python 3.10+。基于 ChromaManga 核心算法，通过 Model Context Protocol (MCP) 暴露给 AI 助手（Claude Desktop / Kiro），使其可用自然语言对话驱动漫画上色全流程。

**核心能力：**

- **区域提示词单次推理**：背景 + 每个角色独立 prompt，通过 `RegionalAttnProcessor` 将不同文本嵌入路由到 SDXL UNet 交叉注意力的不同空间区域，一次推理完成多角色精准上色
- **CLIP ReID 自动角色识别**：`utils/character_reid.py` 用 CLIP ViT-L/14 从 `data/refs/<name>/` 参考图提取特征构建模板，匹配 YOLO 检出的人物区域，分配对应 LoRA + 触发词 + 角色 prompt
- **ControlNet 线稿引导**：`controlnet_mode = "union" | "scribble"`，高保真度保留原始线条
- **对话框文字保护**：YOLO 气泡检测 → 椭圆 mask + inpaint 去噪 → 印回原始黑色文字
- **Real-ESRGAN 4x 超分**：可选，`upscale_enabled` 开启
- **任务级步骤管理**：`Task.STEPS_ORDER` 七阶段，每步产物缓存在 `task.results`，支持 `reset_to_step` 任意回退重跑

**数据流：**

```
MCP 客户端（Claude Desktop / Kiro）
    ↓ 自然语言
MCP Server (mcp_server/chromamanga_mcp_server.py, stdio 协议, 15 个工具)
    ↓ 启动时 build_pipeline() 装配 SDXL+ControlNet+LoRA，全局单例复用
TaskManager → Task（按 STEPS_ORDER 驱动）
    ↓
core/colorize.py::colorize_regional()
  1. 加载图 + 分辨率校准（min_resolution / max_infer_resolution）
  2. YOLO 气泡检测 → 椭圆 mask → 提取文字灰度
  3. 线稿提取（utils/preprocess.py，union=LineartAnime / scribble=adaptiveThreshold）
  4. YOLO 人物检测 → 裁剪 crop
  5. CLIP ReID 匹配模板 → 为每个人物分配 prompt+LoRA 触发词
  6. build_region_masks()：latent 空间背景 mask + 每人 bbox mask
  7. encode_prompts()：背景 + 各区域 prompt 各自编码
  8. set_regional_attn()：改写 UNet cross-attn processor
  9. pipe(...) SDXL+ControlNet 单次推理
 10. reset_attn() 还原 processor（**必须**，否则脏状态残留）
 11. blend_lineart 叠加原始线稿 + Real-ESRGAN 超分到原图尺寸
 12. stamp_text 还原气泡文字
    ↓
data/outputs/colored/{stem}_colored.png + debug/{task_id}_colored/*.png
```

**目录结构：**

| 路径 | 职责 |
|---|---|
| `core/colorize.py` | 主上色管线：`build_pipeline()` 装配模型+LoRA+ReID，`colorize_regional()` 端到端执行；包含气泡 inpaint/stamp、区域 mask 构建、prompt 批量编码 |
| `core/config.py` | 配置集中：`base_model_id` / `controlnet_mode` / `lora_configs` / `reid_*` / 推理参数 / 气泡与人物 YOLO 模型路径 |
| `core/task_manager.py` | `Task`（七阶段 `STEPS_ORDER` + `results` 字典 + `reset_to_step`）、`TaskManager`（UUID 索引任务） |
| `utils/preprocess.py` | 去噪策略（none/bilateral/median/nlmeans/...）+ 线稿提取（lineart_anime / scribble） |
| `utils/postprocess.py` | `blend_lineart` 线稿叠加融合 |
| `utils/character_reid.py` | `CharacterReID`：CLIP 单例 + 参考图模板缓存（`reid_templates.npy`） + 余弦相似度匹配 |
| `utils/regional_attention.py` | `RegionalAttnProcessor`：替换 UNet 所有 cross-attn 处理器，按 latent mask 路由不同区域的 encoder_hidden_states，支持 CFG（batch=2） |
| `mcp_server/chromamanga_mcp_server.py` | MCP Server 入口：`list_tools()` 注册工具 + `call_tool()` 路由 + 每个 `handle_xxx` 实现；stdio 协议，日志走 stderr |
| `mcp_server/mcp_config_example.json` | MCP 客户端侧配置样板 |
| `agent/config.py` | Agent 侧配置：LLM 凭据（`LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL`）、MCP 启动命令、上传目录、FastAPI 监听地址；从 `.env.agent` 懒加载 |
| `agent/mcp_bridge.py` | `MultiServerMCPClient` 封装：stdio 启动 MCP Server，把工具转成 LangChain Tool（懒加载 + 缓存） |
| `agent/graph.py` | `create_react_agent` 构建 ReAct Agent；`SYSTEM_PROMPT` 硬编码了工具名和阶段顺序，是 Agent 行为的单点真相 |
| `agent/server.py` | FastAPI：托管 `frontend/` + `/upload` 接收图 + `/image` 回任意绝对路径图 + `/ws` WebSocket 流式推送（stream_mode="updates"）|
| `run_agent.py` | Agent 入口：validate 配置 → 启 uvicorn → lifespan 内预热 MCP |
| `frontend/index.html` `styles.css` `app.js` | 本地 chat UI：暗色玻璃拟态 + 紫青渐变 + 工具卡可视化 + 拖拽/粘贴/按钮三路上传 + 灯箱 |
| `.env.agent.example` | LLM 凭据样板；复制为 `.env.agent` 并填 key 后 `run_agent.py` 自动加载 |
| `scripts/batch_colorize.py` | 批处理 `data/inputs/` 所有图片 |
| `scripts/experiments/` | 去噪 / 线稿方法对比实验（非主流程） |
| `tests/verify_integration.py` | 集成验证（模型路径、导入） |
| `tests/check_mcp_setup.py` | MCP 依赖/配置检查 |
| `tests/test_mcp_server.py` | MCP Server 基础测试 |
| `tests/test_full_pipeline.py <image>` | 端到端完整流程测试（15-20 分钟） |
| `test_all_imports.py` / `test_import.py` / `test_config_paths.py` | 根目录冒烟测试 |
| `data/{inputs,outputs,loras,models,refs}/` | 数据目录；`models_dir` / `loras_dir` 在 `config.py` 中被重定向到**仓库外**共享路径（`E:\code\graduationProject\Theresa\...`） |

## MCP 工具清单（当前 15 个）

| 分组 | 工具 | 用途 |
|---|---|---|
| 任务管理 | `create_task` / `get_task_status` / `get_task_result` | 基于 `image_path` 建任务，查询进度和 `STEPS_ORDER` 完成情况，拉取最终图+中间产物 |
| 流程步骤 | `extract_lineart` / `detect_persons` / `identify_characters` / `detect_bubbles` / `generate_masks` / `run_inference` / `postprocess` | 严格对应 `Task.STEPS_ORDER` 七阶段，顺序执行 |
| 配置管理 | `get_config` / `update_inference_params` / `update_prompt` | 读取当前 `core.config`；运行时调整 `num_inference_steps`/`guidance_scale`/`controlnet_scale`/`seed`；按 `character_index` 改 LoRA prompt 或背景/负面 prompt |
| 实用工具 | `analyze_image` / `reset_to_step` | 分析某任务某阶段图的亮度/对比度/饱和度；回退任务到任意阶段重跑 |

## 关键命令

```bash
pip install -r requirements.txt
pip install -r mcp_server/mcp_requirements.txt
pip install -r agent_requirements.txt              # 仅跑 Agent + 前端时需要

# ── 三种运行形态 ──
python run_agent.py                                # Agent + 前端（推荐），浏览器开 http://127.0.0.1:8000
python mcp_server/chromamanga_mcp_server.py        # 裸 MCP Server（stdio，供 Claude Desktop/Kiro 拉起）
python core/colorize.py --input data/inputs/x.png  # CLI 直跑端到端
python scripts/batch_colorize.py                   # 批处理 data/inputs/ 全部图片

# ── 体检与测试 ──
python tests/check_mcp_setup.py                    # 先跑：依赖/配置体检
python tests/verify_integration.py                 # 集成验证
python tests/test_mcp_server.py                    # MCP 基础测试
python tests/test_full_pipeline.py <image>         # 端到端流程（15-20 分钟，需 GPU）
python test_all_imports.py                         # 导入冒烟测试（最轻量）
```

## 运行模式

| 模式 | 触发方式 | 适用场景 |
|---|---|---|
| **Agent + Web UI**（推荐） | `python run_agent.py` → 浏览器访问 `http://127.0.0.1:8000` | 本地自用、可视化工具调用过程、拖拽上传、图片灯箱预览 |
| **MCP Server 裸跑** | 外部客户端（Claude Desktop / Kiro）按 `mcp_config_example.json` 拉起 | 接入其他 AI 助手生态 |
| **CLI 单张** | `python core/colorize.py --input <path>` | 离线调试、单图端到端 |
| **批处理** | `python scripts/batch_colorize.py` | `data/inputs/` 目录批量处理，复用同一 pipeline |

### Agent 模式工作机制

```
浏览器 (frontend/)
  ↕ WebSocket /ws  +  POST /upload  +  GET /image?path=...
FastAPI (agent/server.py, lifespan 内预热 Agent)
  ↕ LangGraph create_react_agent（工具循环，SYSTEM_PROMPT 指导）
  ↕ langchain-mcp-adapters (stdio 子进程)
MCP Server (mcp_server/chromamanga_mcp_server.py, 启动时 initialize_models)
  → ChromaManga core pipeline
```

- **首次启动**：`run_agent.py` → `lifespan` → `build_agent()` → `MultiServerMCPClient` 拉 MCP 子进程 → MCP `initialize_models()` 加载 SDXL（30~60 秒）→ 加载工具列表 → Agent 就绪
- **消息流**：`stream_mode="updates"` 逐节点推送，每条 `AIMessage` 拆成 `ai` + `tool_call` 帧，每条 `ToolMessage` 自动扫描文本里的图片路径附带在 `tool_result.images` 里
- **会话状态**：单次 WebSocket 连接内 `history` 常驻内存；`{"type":"reset"}` 消息可清空；刷新页面即新连接（history 为空）；目前未做跨连接持久化

### HTTP / WebSocket 端点

| 端点 | 用途 |
|---|---|
| `GET /` / `/styles.css` / `/app.js` | 静态前端 |
| `GET /config` | 前端启动时拉取环境：`{mode, model, tool_total}`，用于 Header MOCK/REAL 徽章和进度条分母 |
| `POST /upload` | 前端上传图到 `data/inputs/upload_{ts}.ext`，返回绝对路径 |
| `GET /image?path=<abs>` | 按绝对路径返回任意图片（**路径未白名单，仅本地用**） |
| `WS /ws` | 双向流：上行 `{content, image_path}` 或 `{type:"reset"}`；下行 `ai` / `tool_call` / `tool_result` / `done` / `reset_done` / `error` |

### 前端 UI 状态机（app.js）

- Header 三件套：MOCK/REAL 徽章（`/config` 驱动）、进度 `N / 7 · 工具中文名`（由 `STEP_INDEX` 映射驱动）、status 气泡（`LLM 思考中...` / `调用 XXX...` / `已就绪` / `执行出错` 四态）
- `addToolCard` 时推进进度和更新 status；`tool_result` 时回到 "LLM 思考中"；`done` 时回 "已就绪"
- `postprocess` 工具完成后 `appendFinalResult()` 追加一张"🎉 上色完成"大图卡（含下载按钮 + 对比原图按钮）
- 清空按钮：`ws.send({type:"reset"})` → 后端清 history → 下行 `reset_done` → 前端 `resetChatDOM()` 恢复 welcome 屏

### 参数侧栏（参数抽屉）

- Header "参数调节" 按钮 → 右侧 `.param-drawer` 抽屉
- 三滑块：`guidance_scale` / `num_inference_steps` / `controlnet_scale`；一数字框：`seed`（🎲 一键随机）
- 四预设：`vivid` / `soft` / `detailed` / `default`（定义在 `PARAM_PRESETS`）
- **"应用并重跑"机制**：前端不直接调 MCP 工具，而是构造一条结构化中文指令作为用户消息发给 Agent（"请按以下参数重跑 ...  请调用 update_inference_params ..."）。Agent 读指令 → 依次 tool_call `update_inference_params` → `reset_to_step('inference')` → `run_inference` → `postprocess` → `get_task_result`。这样调参过程**完全可见**（chat 里用户能看到完整指令 + Agent 的每一步），不走暗箱
- 前端追踪的会话状态：`currentTaskId`（从 `tool_call.args.task_id` 提取）、`originalImagePath`（发送前记录）、`finalImagePath`（从 postprocess 的 `tool_result.images` 最后一个提取）；任一为空时"应用"按钮禁用

### 对比视图（swipe slider）

- 最终图卡片 `.final-card` 头部 "对比原图" 按钮 → `toggleCompareView()`
- 实现：`.compare-base` 铺底 + `.compare-after` 用 `clip-path: inset(0 0 0 X%)` 切割 + `.compare-handle` 垂直分割线
- 拖动：handle `mousedown/touchstart` → 全局 `mousemove/touchmove` 同步 `handle.style.left` 和 `after.style.clipPath`；`mouseup/touchend` 停
- 依赖 `originalImagePath`（用户上传时记录），所以对比视图**必须在有上传的会话里才生效**；没有原图时 alert 提示

### `AGENT_MCP_MODE` 两档切换

`.env.agent` 里的开关，决定 `agent/mcp_bridge.py` 加载哪套工具：

| 值 | 行为 | 适用 |
|---|---|---|
| `real`（默认） | 按上图启动真 MCP 子进程，跑 SDXL 推理 | 有 GPU + 模型文件齐全的机器 |
| `mock` | 加载 `agent/mock_tools.py`，15 个同签名的假工具：`run_inference` 用 `asyncio.sleep(3)` 模拟耗时；各阶段产物**从 `examples/c1a670e96.../` 拷贝真实上色效果图**（线稿、气泡检测、ReID、区域 mask、彩色结果等），缺失时回退到用户原图；产物落盘 `data/outputs/colored/debug/mock-xxx_colored/` | 只有开发机、只想跑前端/Agent 循环 |

Mock 模式下的"完美演示路径"：
- welcome 屏点 **"✨ 用示例图体验"** chip → 前端 `POST /example-upload` → 后端把 `examples/.../00_input.png` 复制到 `data/inputs/example_{ts}.png` → 前端自动填"帮我上色" + 发送 → Agent 走完整流程，每个工具产物都是对应 examples 真图
- 由于原图和各阶段产物来自同一个 task 的真实运行结果，**对比视图**左右完全对齐（黑白原图 ↔ 彩色最终图）
- 新增 `stage → examples 图` 的映射在 `mock_tools.py::_STAGE_SOURCE`，修改 examples 目录时同步这个表

- Mock 模式下 `validate()` 不再检查 `MCP_SERVER_SCRIPT` 存在性
- 切换模式**必须重启** `run_agent.py`（工具列表 `_tools` 是进程级缓存）
- Mock 的工具名 / 参数 / 返回风格**必须**跟真 MCP 对齐；真 MCP 改签名时 `mock_tools.py` 也要同步（列入本文件"文档/代码一致性规则"的触发条件）

## 本项目关键约束

- **GPU 必需**：12GB+ VRAM（RTX 3060 Ti 或更好）；CPU 回退理论可用但极慢，不作为支持路径
- **模型/LoRA 外置**：`core/config.py` 中 `models_dir` / `loras_dir` 硬编码指向 `E:\code\graduationProject\Theresa\models|loras`（复用原 ChromaManga 项目），**不在本仓库内**。基础模型、ControlNet、YOLO（人物/气泡）、LoRA、ReID 模板缓存均从该路径加载；换机器必须改这两个常量或改为环境变量
- **`image_path` 必须绝对路径**：MCP 工具参数要求绝对路径；客户端不保证 CWD
- **`downscale_before_infer` 隐含约束**：推理侧长边被限制在 `max_infer_resolution=1024`，人物 bbox 会按比例缩放到推理分辨率（`colorize.py` 中的 `persons_orig` vs `persons`）；新增涉及 bbox 的阶段时要注意这一对坐标系的转换
- **OOM 处理**：显存紧张时降 `max_infer_resolution` 或启用 `downscale_before_infer=True`；xformers 可选依赖，安装后自动启用加速
- **Agent 侧秘密**：`.env.agent` 必须 gitignore（已配置），示例放 `.env.agent.example`；LLM 凭据**禁止**硬编码到 `agent/config.py`
- **Agent 工具名耦合**：`agent/graph.py::SYSTEM_PROMPT` 里写死了所有 MCP 工具名和阶段顺序。若改 MCP 工具，必须同步改 SYSTEM_PROMPT，否则 Agent 会调到不存在的工具或漏调必需步骤

## 通用编程原则（基于第一性原理）

1. **命名即文档** — 名称准确反映用途，禁 `data`/`temp`/`x1`；读者应能从名称直接理解"是什么"和"做什么"
2. **函数单一职责** — 一个函数只做一件事；若需解释"它做了 A，然后做 B，再处理 C"，应拆分
3. **优先不可变** — 默认只读/冻结数据类，副作用限最小范围；可变状态是意外复杂性的主要来源
4. **最小化作用域** — 局部优于全局，私有优于公开，避免全局状态
5. **显式优于隐式** — 禁魔法数字、隐式转换、隐式依赖；所有重要常量、配置、依赖关系显式声明
6. **扁平化结构** — guard clauses 减嵌套，组合代替继承，避免过深的回调链
7. **注释解释"为什么"** — 不注释"是什么"；业务背景、权衡决策、特殊处理的原因才是注释的价值
8. **谨慎设计模式** — 真正降低当前问题复杂性时才用，不为"用模式"而提前引入抽象
9. **错误处理明确** — 不吞异常，不抛通用异常，提供足够上下文；遵循语言/框架的错误处理惯例
10. **可测试性优先** — 避免静态依赖、隐藏输入，提倡依赖注入；难以测试 = 耦合度高 + 职责不单一
11. **代码对称一致** — 相似问题用相似结构，避免多种方式解决同一类问题
12. **质疑惯例** — 从本质推导，不盲从；当前上下文中这个惯例是否真的必要？
13. **关键决策留痕** — 决策分支加诊断日志，使运行时决策链可事后追溯

## Python 约定（本项目）

- **类型注解必写**：函数签名全部标注；Python 3.10+，优先 `str | None` 于 `Optional[str]`
- **配置集中**：所有模型路径 / 推理参数 / prompt 模板仅在 `core/config.py` 声明；其他模块 `from core import config` 后按属性读取，**禁止**在业务代码里散落绝对路径或魔法数字
- **异步边界清晰**：
  - MCP SDK 约束：`list_tools()` / `call_tool()` / `handle_xxx()` 必须为 `async`；SDK 通过 stdio 驱动事件循环
  - 工具实现里的 GPU 推理（`pipe(...)`、CLIP、cv2）是**阻塞的同步 IO**，当前单任务串行场景下可直接调用；若未来并发化，须用 `asyncio.to_thread` 包裹
  - 不要在 async 里做 `time.sleep`，用 `asyncio.sleep`
- **全局单例谨慎**：`pipeline` / `reid` / `task_manager` 是 MCP Server 模块级全局变量，仅在 `initialize_models()` 中装配一次；切勿在工具实现中重载模型
- **注意力处理器生命周期**：`set_regional_attn()` 与 `reset_attn()` 必须成对出现；异常路径也要 `reset_attn()`，否则下次推理沿用脏状态
- **外部依赖可替换**：测试中通过 monkeypatch `pipeline` / `reid` 为假对象；避免在函数签名里硬绑具体类
- **数据模型**：任务状态用 `core.task_manager.Task`，中间结果存 `task.results[step_name]`；别在日志/返回值里传裸 dict

## 日志规范

- **MCP Server**：stdio 协议下 stdout 被 MCP 消息占用，**所有日志必须写 stderr**（`print(..., file=sys.stderr)` 或标准 `logging` 默认配置即可）。违反这条会破坏 MCP 通信
- **新代码统一用 `logging`**：每个模块 `logger = logging.getLogger(__name__)`；`print` 仅保留在 `core/colorize.py`、`scripts/*.py` 等 CLI 入口中的用户反馈输出（现存代码暂不强制改造）
- **日志等级**：
  - `INFO` — 阶段开始/结束、关键决策（ReID 命中哪个角色、选择的推理分辨率、检出几个人物/气泡）
  - `WARNING` — 降级分支（ReID 相似度未达阈值回退到默认 prompt、未配置 `bubble_yolo_model` 跳过气泡）
  - `ERROR` — 不可恢复（模型加载失败、CUDA OOM）
  - `exception(...)` — `except` 块内自动附带堆栈
- **格式**：占位参数而非 f-string，便于懒格式化
  ```python
  logger.info("ReID Person%d: %s similarity=%.4f", i + 1, trigger, sim)
  logger.warning("bubble_yolo_model 未配置，跳过气泡文字保护")
  ```
- **关键决策留痕**：每个阶段入口记录 `task_id` + 核心输入尺寸；出口记录该阶段的产物摘要（bbox 数量、mask 尺寸、耗时），便于事后比对 `reset_to_step` 的行为差异

## 复合任务分解（P1）

当用户请求包含 **2 个以上并列目标**时（如"分析 X + 优化 Y + 评估 Z"）：

1. **先分解**为独立原子任务，每个有明确的验证标准，逐个执行
2. **不跨任务并行**：当前步骤验证通过后再开始下一步，确保每步可独立确认
3. 修改多模块时先给出涉及文件清单，用户确认后再动手

## Agent Skills 设计原则

**核心纪律：沉淀。** 重复任务→沉淀为 Skill；有效判断→沉淀决策过程；发现模式→沉淀让系统识别。

| # | 原则 | 要求 |
|---|------|------|
| 1 | **食谱，非命令** | Skill 定义参数化流程（输入、步骤、输出格式），而非一次性硬编码指令 |
| 2 | **教思考，非结论** | 教授权衡、质疑、考虑替代解释的方法，不预设结论 |
| 3 | **判断↔计算边界显式化** | 显式标记哪些步骤是判断（让 AI 思考），哪些是计算（调用工具） |
| 4 | **通读一切再综合** | 读取全部相关文档后综合，不预过滤 |
| 5 | **正确时刻加载正确文档** | 200 行指针优于 2 万行指令 |
| 6 | **智能向上，执行向下** | Skill（流程+判断）→ Harness（纤薄编排）→ Tool（快速可靠单一功能） |
| 7 | **快而窄 > 慢而通用** | 每个工具做一件事，一秒内完成，不解释不决策 |
| 8 | **追逐"还行"响应** | 聚焦 lukewarm 输出：读反馈→找差距→改 Skill→验证改善 |
| 9 | **写一次，永远运行** | 如果必须要求两次，就是失败。能沉淀为 Skill 的立即沉淀 |
| 10 | **相同流程，不同世界** | 参数提供世界（数据/标准），Skill 提供流程（判断/步骤） |
