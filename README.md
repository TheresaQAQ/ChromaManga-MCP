# ChromaManga MCP Server

<div align="center">

**AI-Powered Manga Colorization Service via Model Context Protocol**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-1.0-blue.svg)](https://modelcontextprotocol.io/)

[English](README.md) | [简体中文](README_CN.md)

[Features](#features) • [Quick Start](#quick-start) • [MCP Tools](#mcp-tools) • [Usage Examples](#usage-examples) • [Architecture](#architecture)

</div>

---

## Project Overview

This project is built upon [ChromaManga](https://github.com/TheresaQAQ/ChromaManga), wrapping its powerful manga colorization capabilities as an **MCP (Model Context Protocol) service**, enabling AI assistants (like Claude, Kiro) to perform manga colorization tasks through natural language conversations.

### What is MCP?

MCP is an open protocol that allows AI applications to seamlessly integrate with external tools and data sources. Through MCP, AI assistants can:
- Invoke local tools and services
- Access file systems and databases
- Execute complex multi-step tasks

---

## Features

### Core Colorization Capabilities
🎨 **Regional Prompt Control** - Different prompts for background and each character, precise colorization in a single inference pass

🤖 **Automatic Character Recognition** - CLIP-based ReID automatically identifies characters and applies matching LoRA styles

🎯 **ControlNet Lineart Guidance** - Preserves original manga lineart structure with high fidelity

💬 **Speech Bubble Protection** - Automatically detects and preserves text in speech bubbles

⚡ **Real-ESRGAN Super-Resolution** - Optional 4x super-resolution for high-quality output

### MCP Service Features
🔧 **16 Tool Functions** - Complete pipeline from task creation to post-processing

📊 **Task State Management** - Real-time query of task progress and intermediate results

🔄 **Pipeline Rollback** - Support for rolling back to any step and re-executing

⚙️ **Dynamic Parameter Adjustment** - Modify prompts and inference parameters at runtime

🤝 **AI Conversation Integration** - Use through natural language in Kiro, Claude Desktop, and other clients

---

## Quick Start

### Requirements
- Python 3.10+
- CUDA-capable GPU with 12GB+ VRAM (recommended)
- 20GB+ free disk space for models

### Installation

```bash
# 1. Clone repository
git clone https://github.com/YourUsername/ChromaManga-MCP.git
cd ChromaManga-MCP

# 2. Install core dependencies
pip install -r requirements.txt

# 3. Install MCP dependencies
pip install -r mcp_server/mcp_requirements.txt

# 4. Models will be downloaded automatically on first run
# Or manually download to data/models/ directory:
# - Base model: Illustrious-XL
# - ControlNet: xinsir/controlnet-union-sdxl-1.0
# - YOLO models: anime person detection + bubble detection
```

### Configure MCP Client

#### Using with Kiro

Edit `~/.kiro/settings/mcp.json`:

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

#### Using with Claude Desktop

Edit Claude Desktop configuration file:

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

### Verify Installation

```bash
# Run integration tests
python tests/verify_integration.py

# Test full pipeline (takes 15-20 minutes)
python tests/test_full_pipeline.py data/inputs/manga.png
```

---

## MCP Tools

This service provides 16 tool functions covering the complete manga colorization pipeline:

### Task Management (3 tools)
- `create_task` - Create a new colorization task
- `get_task_status` - Query task status and progress
- `get_task_result` - Get final results and intermediate outputs

### Pipeline Steps (7 tools)
- `extract_lineart` - Extract lineart for ControlNet guidance
- `detect_persons` - Detect person locations in the image
- `identify_characters` - Identify characters using CLIP ReID
- `detect_bubbles` - Detect speech bubbles to protect text areas
- `generate_masks` - Generate region masks
- `run_inference` - Execute SDXL + ControlNet inference (10-20 minutes)
- `postprocess` - Post-processing: blend lineart, restore text, super-resolution

### Configuration Management (3 tools)
- `get_config` - Get current configuration
- `update_inference_params` - Adjust inference parameters (steps, CFG, seed, etc.)
- `update_prompt` - Modify character or background prompts

### Utility Functions (3 tools)
- `analyze_image` - Analyze image quality metrics
- `reset_to_step` - Roll back to a specific step and re-execute

---

## Usage Examples

### Example 1: Complete Colorization Pipeline

In your AI assistant, type:

```
Please colorize this manga: /path/to/manga.png

Execute the complete pipeline:
1. Create task
2. Extract lineart
3. Detect persons
4. Identify characters
5. Detect bubbles
6. Generate masks
7. Run inference
8. Apply post-processing
```

The AI will automatically call all tools to complete the colorization and tell you where the result is saved.

### Example 2: Adjust Colors

```
User: Sagiri's hair is too dark, make it brighter

AI actions:
1. get_config - View current configuration
2. update_prompt - Modify Sagiri's prompt, add "bright, light, vibrant"
3. reset_to_step - Roll back to inference step
4. run_inference - Regenerate
5. postprocess - Apply post-processing
```

### Example 3: Adjust Parameters

```
User: Colors are too pale, make them more vibrant

AI actions:
1. analyze_image - Analyze current result
2. update_inference_params - Increase guidance_scale (6.0 → 7.0)
3. reset_to_step - Roll back to inference
4. run_inference - Regenerate
```

### Example 4: Check Progress

```
User: Check status of task abc-123

AI calls get_task_status to display:
- Current status: processing
- Progress: 75% (6/8 steps)
- Current step: run_inference
- Completed: lineart_extraction, detect_persons, ...
```

---

## Architecture

### Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│ MCP Client (Kiro / Claude Desktop)                              │
│   • User interacts through natural language                     │
│   • AI assistant understands intent and calls MCP tools         │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ MCP Server (chromamanga_mcp_server.py)                          │
│   • 16 tool functions                                           │
│   • Task state management                                       │
│   • Dynamic parameter adjustment                                │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ ChromaManga Core Engine                                         │
│                                                                 │
│  1. Input Processing                                            │
│     • Load B&W manga image                                      │
│     • Extract lineart (ControlNet-compatible)                   │
│     • Detect speech bubbles → extract text mask                 │
│                                                                 │
│  2. Character Detection & Recognition                           │
│     • YOLO person detection → bounding boxes                    │
│     • CLIP ReID → match to character templates                  │
│     • Assign region-specific prompts                            │
│                                                                 │
│  3. Regional Attention Inference                                │
│     • Build spatial masks (background + person regions)         │
│     • Encode all region prompts                                 │
│     • Inject RegionalAttnProcessor into UNet                    │
│     • Single-pass SDXL + ControlNet inference                   │
│     • Different prompts routed to different regions             │
│                                                                 │
│  4. Post-processing                                             │
│     • Blend original lineart (preserve line clarity)            │
│     • Restore speech bubble text (inpaint + stamp)              │
│     • Real-ESRGAN 4x super-resolution                           │
│     • Save final result + debug visualizations                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Technologies

- **Model Context Protocol (MCP)** - AI tool integration protocol
- **Stable Diffusion XL** - High-quality image generation
- **ControlNet (Lineart)** - Preserves manga line structure
- **Regional Attention** - Custom attention processor for spatial prompt routing
- **CLIP ReID** - Character identification using visual similarity
- **YOLO** - Person and speech bubble detection
- **Real-ESRGAN** - Anime-optimized super-resolution

---

## Project Structure

```
ChromaManga-MCP/
├── core/                    # Core algorithm modules
│   ├── colorize.py          # Main colorization logic
│   ├── config.py            # Configuration management
│   └── task_manager.py      # Task manager
│
├── mcp_server/              # MCP server
│   ├── chromamanga_mcp_server.py  # MCP main service
│   ├── mcp_requirements.txt       # MCP dependencies
│   └── mcp_config_example.json    # Configuration example
│
├── utils/                   # Utility functions
│   ├── preprocess.py        # Lineart extraction and denoising
│   ├── postprocess.py       # Lineart blending
│   ├── character_reid.py    # CLIP character recognition
│   └── regional_attention.py # Regional attention processor
│
├── tests/                   # Test files
│   ├── test_mcp_server.py   # MCP basic tests
│   ├── test_full_pipeline.py # Full pipeline test
│   └── verify_integration.py # Integration verification
│
├── data/                    # Data directory
│   ├── inputs/              # Input images
│   ├── outputs/             # Output results
│   ├── loras/               # LoRA models
│   ├── models/              # Pre-trained models
│   └── refs/                # Character reference images (for ReID)
│
├── docs/                    # Documentation
│   ├── MCP_SERVER_README.md # MCP detailed documentation
│   ├── QUICK_START.md       # Quick start guide
│   └── architecture/        # Architecture design docs
│
└── requirements.txt         # Python dependencies
```

---

## Parameter Tuning Guide

### Color-Related
| Issue | Solution | Parameter Adjustment |
|-------|----------|---------------------|
| Colors too pale | Increase prompt adherence | `guidance_scale: 6.0 → 7.0` |
| Colors oversaturated | Decrease prompt adherence | `guidance_scale: 6.0 → 5.0` |
| Specific character color wrong | Modify character prompt | `update_prompt` add color keywords |

### Quality-Related
| Issue | Solution | Parameter Adjustment |
|-------|----------|---------------------|
| Insufficient detail | Increase inference steps | `num_inference_steps: 20 → 25` |
| Lineart not clear | Increase ControlNet strength | `controlnet_scale: 1.1 → 1.2` |
| Too rigid | Decrease ControlNet strength | `controlnet_scale: 1.1 → 0.9` |
| Lineart too heavy | Decrease blend strength | `blend_alpha: 0.15 → 0.1` |

### Background-Related
| Issue | Solution | Example Prompt |
|-------|----------|----------------|
| Background too busy | Simplify background prompt | `"simple white background, minimal"` |
| Background too plain | Enrich background prompt | `"detailed indoor scene, furniture, ..."` |

---

## Character Auto-Recognition

Place reference images in `data/refs/character_name/` directory:

```
data/refs/
├── sagiri/
│   ├── ref1.png
│   ├── ref2.jpg
│   └── ref3.png
└── another_character/
    └── ref1.png
```

The system will automatically:
1. Extract CLIP features from reference images
2. Build character templates (cached in `data/models/reid_templates.npy`)
3. Match detected persons to templates
4. Apply corresponding LoRA and prompts

---

## System Requirements

- **GPU**: NVIDIA GPU with 12GB+ VRAM (RTX 3060 Ti or better)
- **RAM**: 16GB+ system memory
- **Storage**: 20GB+ for models and cache
- **OS**: Windows 10/11, Linux, macOS (with CUDA support)

---

## Troubleshooting

### MCP-Related

**MCP Server won't start**
- Check if configuration file path is correct
- Run `python tests/verify_integration.py` to verify
- Check client log output

**Tool call failed**
- Confirm task ID is correct
- Check if prerequisite steps are completed
- Review error message and follow instructions

### Performance-Related

**Out of Memory (OOM)**
- Reduce `max_infer_resolution` in `core/config.py`
- Enable `downscale_before_infer = True`
- Close other GPU applications

**Slow inference**
- Ensure CUDA is available: `torch.cuda.is_available()`
- Install xformers: `pip install xformers`
- Use fewer inference steps (15-20)

### Quality-Related

**Poor colorization quality**
- Adjust `controlnet_scale` (try 0.9-1.2)
- Tune `guidance_scale` (try 5.5-7.5)
- Check if LoRA is compatible with base model
- Ensure lineart extraction quality (check debug images)

---

## Output Results

Each task generates complete intermediate results:

```
data/outputs/colored/debug/{task_id}_colored/
├── 00_input.png              # Original input
├── 01_bubble_bbox.png         # Bubble detection
├── 02_lineart.png             # Lineart extraction
├── 02_person_detection.png    # Person detection
├── 03_bboxes_reid.png         # Character identification
├── 04_region_mask.png         # Region masks
├── 05_colored_raw.png         # Raw colorization
├── 06_lineart_blend.png       # Lineart blended
└── 07_final.png               # Final result

data/outputs/colored/
└── {filename}_colored.png     # Main output file
```

---

## Development Guide

### Adding New MCP Tools

1. Add tool definition in `list_tools()`
2. Add routing in `call_tool()`
3. Implement corresponding `handle_xxx()` function

### Debugging MCP Server

```bash
# Run MCP Server directly
python mcp_server/chromamanga_mcp_server.py

# View logs (output to stderr)
python mcp_server/chromamanga_mcp_server.py 2> mcp_server.log
```

### Running Tests

```bash
# Basic functionality tests
python tests/test_mcp_server.py

# Full pipeline test
python tests/test_full_pipeline.py data/inputs/manga.png

# Integration verification
python tests/verify_integration.py
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

This project is built upon the following open-source projects:

- [ChromaManga](https://github.com/TheresaQAQ/ChromaManga) - Original manga colorization project
- [Model Context Protocol](https://modelcontextprotocol.io/) - MCP protocol specification
- [Stable Diffusion XL](https://github.com/Stability-AI/generative-models) by Stability AI
- [ControlNet](https://github.com/lllyasviel/ControlNet) by Lvmin Zhang
- [Illustrious-XL](https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0) by OnomaAI Research
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) by Xintao Wang
- [CLIP](https://github.com/openai/CLIP) by OpenAI
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

---

## Related Documentation

- [MCP Server Detailed Documentation](docs/MCP_SERVER_README.md)
- [Quick Start Guide](docs/QUICK_START.md)
- [API Design Documentation](docs/API_DESIGN.md)
- [Architecture Design Documents](docs/architecture/)

---

## Citation

If you use this project in your research or project, please cite:

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

**Made with ❤️ for manga artists and AI developers**

[Report Issues](https://github.com/YourUsername/ChromaManga-MCP/issues) • [Feature Requests](https://github.com/YourUsername/ChromaManga-MCP/issues) • [Contribute](https://github.com/YourUsername/ChromaManga-MCP/pulls)

</div>
