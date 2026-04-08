# Changelog

## [Unreleased] - 2024-04-08

### Changed - 项目结构重组

#### 目录结构优化
- 创建 `core/` 模块：核心算法（colorize.py, config.py, task_manager.py）
- 创建 `mcp_server/` 模块：MCP 服务器相关文件
- 创建 `tests/` 目录：所有测试文件
- 创建 `scripts/` 目录：工具脚本和实验代码
- 创建 `docs/` 目录：所有文档（架构、论文、开发）
- 创建 `data/` 目录：数据文件（inputs, outputs, loras, models, refs）
- 创建 `examples/` 目录：示例和演示

#### 文件移动
- `colorize.py` → `core/colorize.py`
- `config.py` → `core/config.py`
- `task_manager.py` → `core/task_manager.py`
- `chromamanga_mcp_server.py` → `mcp_server/chromamanga_mcp_server.py`
- `mcp_requirements.txt` → `mcp_server/mcp_requirements.txt`
- `mcp_config_example.json` → `mcp_server/mcp_config_example.json`
- 测试文件 → `tests/`
- 实验脚本 → `scripts/experiments/`
- 文档文件 → `docs/` (按类型分类)
- 数据目录 → `data/` (inputs, outputs, loras, models, refs)

#### 文档更新
- 更新 `README.md` 以反映新的项目结构
- 更新 `.gitignore` 以适配新的目录结构
- 创建 `docs/PROJECT_STRUCTURE.md` 详细说明项目结构
- 创建各模块的 `__init__.py` 文件

#### 删除
- 删除 `config - 副本.py` (重复文件)

### 优势

1. **模块化**: 清晰的模块划分，便于维护和扩展
2. **专业性**: 符合 Python 项目最佳实践
3. **可读性**: 目录结构一目了然
4. **可扩展**: 为 Agent 系统预留空间
5. **文档完善**: 分类清晰的文档结构

### 下一步计划

- [ ] 开发 Agent 系统（agent/ 目录）
- [ ] 开发 FastAPI 后端（backend/ 目录）
- [ ] 开发 Web UI（frontend/ 目录）
- [ ] 完善测试覆盖率
- [ ] 添加 CI/CD 流程

---

## [1.0.0] - 2024-04-01

### Added
- 初始版本发布
- 核心上色算法实现
- MCP 服务器实现
- 基础测试和文档
