/**
 * ChromaManga Agent 前端
 * - WebSocket 连接 /ws，流式接收 ai / tool_call / tool_result / done / error
 * - 文件通过 POST /upload 上传到后端，拿到绝对路径再随消息发给 Agent
 * - 工具调用过程用卡片可视化，可展开查看参数与结果，点击图片灯箱放大
 */
(function () {
  "use strict";

  // ─── DOM ────────────────────────────────────────────────────────
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => Array.from(document.querySelectorAll(s));

  const chat = $("#chat");
  let welcomeEl = $("#welcome");
  const welcomeHTML = welcomeEl ? welcomeEl.outerHTML : "";
  const input = $("#input");
  const btnSend = $("#btn-send");
  const btnUpload = $("#btn-upload");
  const fileInput = $("#file-input");
  const uploadPreview = $("#upload-preview");
  const previewImg = $("#preview-img");
  const uploadPath = $("#upload-path");
  const clearUpload = $("#clear-upload");
  const status = $("#status");
  const statusText = status.querySelector(".status-text");
  const dropzone = $("#dropzone");
  const lightbox = $("#lightbox");
  const lightboxImg = $("#lightbox-img");
  const modeBadge = $("#mode-badge");
  const progressEl = $("#progress");
  const progressFill = $("#progress-fill");
  const progressText = $("#progress-text");
  const btnClear = $("#btn-clear");
  const btnDrawer = $("#btn-drawer");
  const drawer = $("#param-drawer");
  const drawerBackdrop = $("#drawer-backdrop");
  const drawerStatus = $("#drawer-status");
  const btnDrawerClose = $("#drawer-close");
  const btnApply = $("#btn-apply");
  const btnRandomSeed = $("#btn-random-seed");

  // LLM 配置抽屉
  const btnLLM = $("#btn-llm");
  const llmDrawer = $("#llm-drawer");
  const btnLLMClose = $("#llm-drawer-close");
  const btnLLMApply = $("#btn-llm-apply");
  const llmStatus = $("#llm-status");
  const llmBaseUrl = $("#llm-base-url");
  const llmModel = $("#llm-model");
  const llmModelList = $("#llm-model-list");
  const llmApiKey = $("#llm-api-key");
  const llmKeyHint = $("#llm-key-hint");
  const llmPersist = $("#llm-persist");
  const llmCurrentInfo = $("#llm-current-info");

  // 流程步骤名 → 进度索引（1~7）。不在表里的工具不推进进度（config/update/analyze/reset）
  const STEP_INDEX = {
    extract_lineart: 1,
    detect_persons: 2,
    identify_characters: 3,
    detect_bubbles: 4,
    generate_masks: 5,
    run_inference: 6,
    postprocess: 7,
  };
  const TOOL_LABEL = {
    create_task: "创建任务",
    extract_lineart: "提取线稿",
    detect_persons: "检测人物",
    identify_characters: "识别角色",
    detect_bubbles: "检测对话框",
    generate_masks: "生成蒙版",
    run_inference: "SDXL 推理",
    postprocess: "后处理",
    get_task_result: "整理结果",
    get_task_status: "查询状态",
    get_config: "查询配置",
    update_inference_params: "调整参数",
    update_prompt: "修改提示词",
    analyze_image: "分析图像",
    reset_to_step: "回退阶段",
  };
  let progressTotal = 7;
  let currentProgress = 0;

  // ── 会话状态（供参数侧栏 / 对比视图使用） ─────────────────────────
  let currentTaskId = null;
  let originalImagePath = null;  // 用户最近一次上传的图
  let finalImagePath = null;     // postprocess 最近一次产出的最终图

  // ── LLM 服务商预设（数据来源：百炼/DeepSeek 官方文档 2026-04） ────
  const PROVIDER_PRESETS = {
    dashscope: {
      label: "阿里百炼",
      base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1",
      models: [
        // 旗舰系（最强工具调用）
        { id: "qwen3-max", label: "qwen3-max（最新旗舰 · 推荐）" },
        { id: "qwen3-max-preview", label: "qwen3-max-preview（旗舰预览版）" },
        { id: "qwen-max-latest", label: "qwen-max-latest（Max 最新快照）" },
        { id: "qwen-max", label: "qwen-max（Max 稳定版）" },
        // Plus 系（性价比首选）
        { id: "qwen3.5-plus", label: "qwen3.5-plus（最新 Plus · 北京区）" },
        { id: "qwen-plus-latest", label: "qwen-plus-latest（Plus 最新快照）" },
        { id: "qwen-plus", label: "qwen-plus（Plus 稳定 · 默认）" },
        // Flash / Turbo（快、便宜）
        { id: "qwen3.5-flash", label: "qwen3.5-flash（最新 Flash · 北京区）" },
        { id: "qwen-flash", label: "qwen-flash（Flash · 低成本）" },
        { id: "qwen-turbo-latest", label: "qwen-turbo-latest（Turbo 最快）" },
        { id: "qwen-turbo", label: "qwen-turbo" },
        // Coder（编程优化）
        { id: "qwen3-coder-plus", label: "qwen3-coder-plus（编程 · 强）" },
        { id: "qwen3-coder-flash", label: "qwen3-coder-flash（编程 · 快）" },
        // 思考型
        { id: "qwq-plus-latest", label: "qwq-plus-latest（推理 · 慢但更聪明）" },
        { id: "qwq-plus", label: "qwq-plus" },
      ],
      key_hint: "在 dashscope.console.aliyun.com 控制台 → API-KEY 获取",
    },
    deepseek: {
      label: "DeepSeek",
      base_url: "https://api.deepseek.com/v1",
      models: [
        // V4 主力（2026 起取代 V3 chat / R1 reasoner）
        { id: "deepseek-v4-pro", label: "deepseek-v4-pro（V4 旗舰 · 1.6T 参数）" },
        { id: "deepseek-v4-flash", label: "deepseek-v4-flash（V4 默认 · 工具调用稳定）" },
        // Legacy 兼容名（2026/07/24 弃用）
        { id: "deepseek-chat", label: "deepseek-chat（legacy · 已路由到 v4-flash 非思考）" },
        { id: "deepseek-reasoner", label: "deepseek-reasoner（legacy · ⚠ 不支持工具调用）" },
      ],
      key_hint: "在 platform.deepseek.com → API Keys 获取。⚠ deepseek-reasoner 不支持 tool calls，Agent 用不了；推荐 v4-flash 或 v4-pro。",
    },
    custom: {
      label: "自定义",
      base_url: "",
      models: [],
      key_hint: "任意 OpenAI 兼容端点（含 /v1 路径）",
    },
  };
  let currentProvider = "dashscope";
  let serverLLMConfig = null; // 后端当前真实配置，用于回显与判断 has_api_key

  // ── 推理参数状态（滑块 & 预设驱动） ───────────────────────────────
  const DEFAULT_PARAMS = {
    guidance_scale: 6.0,
    num_inference_steps: 20,
    controlnet_scale: 1.1,
    seed: 42,
  };
  const PARAM_PRESETS = {
    vivid:    { guidance_scale: 7.5, num_inference_steps: 22, controlnet_scale: 1.1,  seed: -1 },
    soft:     { guidance_scale: 5.0, num_inference_steps: 20, controlnet_scale: 0.95, seed: -1 },
    detailed: { guidance_scale: 6.5, num_inference_steps: 28, controlnet_scale: 1.2,  seed: -1 },
    default:  { ...DEFAULT_PARAMS },
  };
  const currentParams = { ...DEFAULT_PARAMS };

  // ─── State ──────────────────────────────────────────────────────
  let ws = null;
  let wsReconnectTimer = null;
  let pendingImagePath = null;
  let currentAIBubble = null;   // 当前正在填充的 AI 气泡
  let pendingAIText = "";       // 流式累积的当前气泡文本（用于 marked 重渲染）
  let typingIndicator = null;
  const toolCards = new Map();  // tool_call_id -> card element

  // ─── Markdown (marked 已通过 CDN 引入) ──────────────────────────
  if (window.marked) {
    marked.setOptions({ gfm: true, breaks: true });
  }

  // ─── WebSocket ──────────────────────────────────────────────────
  function connect() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${proto}//${location.host}/ws`);

    ws.addEventListener("open", () => {
      setStatus("ready", "已就绪");
      clearTimeout(wsReconnectTimer);
    });

    ws.addEventListener("close", () => {
      setStatus("error", "连接断开，3 秒后重连...");
      wsReconnectTimer = setTimeout(connect, 3000);
    });

    ws.addEventListener("error", () => {
      setStatus("error", "连接错误");
    });

    ws.addEventListener("message", (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        handleServerMessage(msg);
      } catch (err) {
        console.error("无法解析消息", err, ev.data);
      }
    });
  }

  function setStatus(cls, text) {
    status.className = "status " + cls;
    statusText.textContent = text;
  }

  function setProgress(idx, toolName) {
    if (idx > 0) {
      currentProgress = Math.max(currentProgress, idx);
      progressEl.hidden = false;
      progressFill.style.width = `${(currentProgress / progressTotal) * 100}%`;
      const label = TOOL_LABEL[toolName] || toolName || "";
      progressText.textContent = label
        ? `${currentProgress} / ${progressTotal} · ${label}`
        : `${currentProgress} / ${progressTotal}`;
    }
  }

  function resetProgress() {
    currentProgress = 0;
    progressEl.hidden = true;
    progressFill.style.width = "0%";
    progressText.textContent = `0 / ${progressTotal}`;
  }

  async function loadServerConfig() {
    try {
      const res = await fetch("/config");
      if (!res.ok) return;
      const cfg = await res.json();
      progressTotal = cfg.tool_total || 7;
      progressText.textContent = `0 / ${progressTotal}`;
      if (cfg.mode === "mock") {
        modeBadge.textContent = "🧪 MOCK";
        modeBadge.className = "mode-badge mock";
        modeBadge.title = "Mock 模式 - 使用假工具，不调用真实 MCP Server";
        modeBadge.hidden = false;
      } else {
        modeBadge.textContent = "● REAL";
        modeBadge.className = "mode-badge real";
        modeBadge.title = `真 MCP 模式 - ${cfg.model}`;
        modeBadge.hidden = false;
      }
    } catch (err) {
      console.warn("加载 /config 失败:", err);
    }
  }

  async function loadLLMConfig() {
    try {
      const res = await fetch("/llm-config");
      if (!res.ok) return;
      serverLLMConfig = await res.json();
      // 根据 base_url 自动判断属于哪个预设
      let detected = "custom";
      for (const [key, preset] of Object.entries(PROVIDER_PRESETS)) {
        if (preset.base_url && serverLLMConfig.base_url === preset.base_url) {
          detected = key;
          break;
        }
      }
      currentProvider = detected;
      switchProviderTab(detected, /* fromServer */ true);
      llmBaseUrl.value = serverLLMConfig.base_url || "";
      llmModel.value = serverLLMConfig.model || "";
      llmApiKey.placeholder = serverLLMConfig.has_api_key
        ? `当前: ${serverLLMConfig.api_key_masked}（留空沿用）`
        : "sk-...";
      llmCurrentInfo.innerHTML = serverLLMConfig.has_api_key
        ? `<b>${PROVIDER_PRESETS[detected].label}</b> · ${serverLLMConfig.model}<br>${serverLLMConfig.api_key_masked}`
        : "<b>未配置</b>，请填入 API Key";
    } catch (err) {
      console.warn("加载 /llm-config 失败:", err);
    }
  }

  function switchProviderTab(provider, fromServer = false) {
    currentProvider = provider;
    $$(".provider-tab").forEach((t) => t.classList.toggle("active", t.dataset.provider === provider));
    const preset = PROVIDER_PRESETS[provider];
    if (!preset) return;
    llmKeyHint.textContent = preset.key_hint;

    // 刷新 model datalist
    llmModelList.innerHTML = "";
    preset.models.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = m.label;
      llmModelList.appendChild(opt);
    });

    if (!fromServer) {
      // 用户主动切 tab：填入预设默认值
      llmBaseUrl.value = preset.base_url;
      llmBaseUrl.readOnly = provider !== "custom";
      llmModel.value = preset.models[0]?.id || "";
    } else {
      // 启动加载场景：保留服务端真实值，但让 base_url 可编辑性符合 provider
      llmBaseUrl.readOnly = provider !== "custom";
    }
  }

  async function applyLLMConfig() {
    const base_url = llmBaseUrl.value.trim();
    const model = llmModel.value.trim();
    const api_key = llmApiKey.value.trim();
    const persist = llmPersist.checked;

    if (!base_url || !model) {
      llmStatus.textContent = "Base URL 和 Model 不能为空";
      llmStatus.className = "drawer-status warning";
      return;
    }
    if (!serverLLMConfig?.has_api_key && !api_key) {
      llmStatus.textContent = "首次配置必须填 API Key";
      llmStatus.className = "drawer-status warning";
      return;
    }

    btnLLMApply.disabled = true;
    llmStatus.textContent = "应用中...";
    llmStatus.className = "drawer-status";

    try {
      const res = await fetch("/llm-config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ base_url, model, api_key, persist }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || JSON.stringify(data));

      llmStatus.textContent = `✓ 已切换到 ${PROVIDER_PRESETS[currentProvider].label} / ${data.model}` + (data.persisted ? "（已写入 .env.agent）" : "");
      llmStatus.className = "drawer-status";

      // 刷新当前配置展示
      llmApiKey.value = "";
      await loadLLMConfig();
      setTimeout(() => closeLLMDrawer(), 1200);
    } catch (err) {
      llmStatus.textContent = `✗ 失败: ${err.message}`;
      llmStatus.className = "drawer-status warning";
    } finally {
      btnLLMApply.disabled = false;
    }
  }

  function openLLMDrawer() {
    llmDrawer.hidden = false;
    drawerBackdrop.hidden = false;
    loadLLMConfig();
    llmStatus.textContent = "";
  }
  function closeLLMDrawer() {
    llmDrawer.hidden = true;
    drawerBackdrop.hidden = true;
  }

  // ─── Incoming ───────────────────────────────────────────────────
  function handleServerMessage(msg) {
    switch (msg.type) {
      case "ai":
        appendAIText(msg.content);
        setStatus("busy", "LLM 思考中...");
        break;
      case "tool_call":
        addToolCard(msg);
        setStatus("busy", `调用 ${TOOL_LABEL[msg.name] || msg.name}...`);
        setProgress(STEP_INDEX[msg.name] || 0, msg.name);
        break;
      case "tool_result":
        updateToolCard(msg);
        setStatus("busy", "LLM 思考中...");
        break;
      case "error":
        appendError(msg.content);
        stopTyping();
        setStatus("error", "执行出错");
        break;
      case "done":
        stopTyping();
        currentAIBubble = null;
        setStatus("ready", "已就绪");
        break;
      case "reset_done":
        resetChatDOM();
        setStatus("ready", "已就绪");
        break;
      default:
        console.warn("未知消息类型", msg);
    }
    scrollToBottom();
  }

  function resetChatDOM() {
    chat.innerHTML = welcomeHTML;
    welcomeEl = $("#welcome");
    bindChipButtons();
    resetProgress();
    toolCards.clear();
    currentAIBubble = null;
    pendingAIText = "";
    currentTaskId = null;
    originalImagePath = null;
    finalImagePath = null;
    updateDrawerStatus();
    stopTyping();
  }

  function removeWelcome() {
    if (welcomeEl && welcomeEl.parentNode) welcomeEl.remove();
  }

  function appendUserMessage(text, imagePath) {
    removeWelcome();
    const el = document.createElement("div");
    el.className = "message user";
    el.innerHTML = `
      <div class="avatar user">你</div>
      <div class="bubble"></div>
    `;
    const bubble = el.querySelector(".bubble");
    if (text) {
      const p = document.createElement("p");
      p.textContent = text;
      bubble.appendChild(p);
    }
    if (imagePath) {
      const img = document.createElement("img");
      img.className = "inline";
      img.src = `/image?path=${encodeURIComponent(imagePath)}`;
      img.alt = "用户上传";
      img.addEventListener("click", () => openLightbox(img.src));
      bubble.appendChild(img);
    }
    chat.appendChild(el);
  }

  /**
   * 追加 token 级增量到当前 AI 气泡。
   * 后端 messages 流每个 LLM token 调一次 → 前端累积 pendingAIText 后用 marked 重渲染。
   * `currentAIBubble = null` 时（首次/工具卡之后）开新气泡并重置累积器。
   */
  function appendAIText(delta) {
    stopTyping();
    removeWelcome();
    if (!currentAIBubble) {
      const el = document.createElement("div");
      el.className = "message ai";
      el.innerHTML = `
        <div class="avatar ai">C</div>
        <div class="bubble"></div>
      `;
      chat.appendChild(el);
      currentAIBubble = el.querySelector(".bubble");
      pendingAIText = "";
    }
    pendingAIText += delta;
    const rendered = window.marked
      ? marked.parse(pendingAIText)
      : `<p>${escapeHtml(pendingAIText)}</p>`;
    currentAIBubble.innerHTML = rendered;
    currentAIBubble.querySelectorAll("img").forEach((img) => {
      if (img.dataset.bound) return;
      img.dataset.bound = "1";
      img.classList.add("inline");
      img.addEventListener("click", () => openLightbox(img.src));
    });
  }

  function appendError(text) {
    removeWelcome();
    const el = document.createElement("div");
    el.className = "message ai error";
    el.innerHTML = `
      <div class="avatar ai">!</div>
      <div class="bubble"></div>
    `;
    el.querySelector(".bubble").textContent = text;
    chat.appendChild(el);
  }

  function startTyping() {
    if (typingIndicator) return;
    typingIndicator = document.createElement("div");
    typingIndicator.className = "typing";
    typingIndicator.innerHTML = "<span></span><span></span><span></span>";
    chat.appendChild(typingIndicator);
  }

  function stopTyping() {
    if (typingIndicator) {
      typingIndicator.remove();
      typingIndicator = null;
    }
  }

  // ─── Tool cards ─────────────────────────────────────────────────
  function addToolCard(msg) {
    stopTyping();
    removeWelcome();
    currentAIBubble = null; // 工具调用后，AI 的下一段文字另起气泡

    // 记录 task_id，供参数侧栏重跑时使用
    if (msg.args && msg.args.task_id) {
      currentTaskId = msg.args.task_id;
      updateDrawerStatus();
    }

    const card = document.createElement("div");
    card.className = "tool-card pending";
    card.innerHTML = `
      <div class="tool-head">
        <div class="tool-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10"/>
            <polyline points="12 6 12 12 16 14"/>
          </svg>
        </div>
        <span class="tool-name"></span>
        <span class="tool-status">执行中</span>
        <svg class="tool-caret" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="6 9 12 15 18 9"/>
        </svg>
      </div>
      <div class="tool-body">
        <div class="tool-section-label">参数</div>
        <div class="tool-content"></div>
      </div>
    `;
    card.querySelector(".tool-name").textContent = msg.name || "(unknown)";
    card.querySelector(".tool-content").textContent =
      JSON.stringify(msg.args || {}, null, 2);
    card.querySelector(".tool-head").addEventListener("click", () => {
      card.classList.toggle("open");
    });
    chat.appendChild(card);
    toolCards.set(msg.id, card);
    startTyping();
  }

  function updateToolCard(msg) {
    const card = toolCards.get(msg.id);
    if (!card) return;

    const isError = /(failed|error|✗)/i.test(msg.content || "");
    card.classList.remove("pending");
    card.classList.add(isError ? "error" : "done");

    const statusEl = card.querySelector(".tool-status");
    statusEl.textContent = isError ? "失败" : "完成";

    const icon = card.querySelector(".tool-icon");
    icon.innerHTML = isError
      ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
           <circle cx="12" cy="12" r="10"/>
           <line x1="15" y1="9" x2="9" y2="15"/>
           <line x1="9" y1="9" x2="15" y2="15"/>
         </svg>`
      : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
           <polyline points="20 6 9 17 4 12"/>
         </svg>`;

    const body = card.querySelector(".tool-body");

    const resultLabel = document.createElement("div");
    resultLabel.className = "tool-section-label";
    resultLabel.textContent = "结果";
    body.appendChild(resultLabel);

    const resultContent = document.createElement("div");
    resultContent.className = "tool-content";
    resultContent.textContent = msg.content || "(空)";
    body.appendChild(resultContent);

      if (msg.images && msg.images.length > 0) {
      const imgLabel = document.createElement("div");
      imgLabel.className = "tool-section-label";
      imgLabel.textContent = `图像产物 (${msg.images.length})`;
      body.appendChild(imgLabel);

      const imgWrap = document.createElement("div");
      imgWrap.className = "tool-images";
      for (const path of msg.images) {
        const img = document.createElement("img");
        img.src = `/image?path=${encodeURIComponent(path)}`;
        img.alt = path.split(/[\\/]/).pop();
        img.loading = "lazy";
        img.addEventListener("click", (e) => {
          e.stopPropagation();
          openLightbox(img.src);
        });
        imgWrap.appendChild(img);
      }
      body.appendChild(imgWrap);
    }

    // 若是 postprocess 工具完成且有产物，额外渲染一个"最终结果"大图卡片
    const toolName = card.querySelector(".tool-name")?.textContent;
    if (!isError && toolName === "postprocess" && msg.images && msg.images.length > 0) {
      appendFinalResult(msg.images[msg.images.length - 1]);
    }
  }

  function appendFinalResult(imagePath) {
    finalImagePath = imagePath;
    updateDrawerStatus();

    const card = document.createElement("div");
    card.className = "final-card";
    const url = `/image?path=${encodeURIComponent(imagePath)}`;
    const filename = imagePath.split(/[\\/]/).pop();
    card.innerHTML = `
      <div class="final-head">
        <span class="final-mark">🎉</span>
        <div class="final-title">
          <strong>上色完成</strong>
          <span class="final-path"></span>
        </div>
        <div class="final-actions">
          <button class="btn-compare" data-action="compare" type="button">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <rect x="3" y="3" width="18" height="18" rx="2"/>
              <line x1="12" y1="3" x2="12" y2="21"/>
            </svg>
            <span>对比原图</span>
          </button>
          <a class="btn-download" download="${escapeHtml(filename)}" href="${url}">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="7 10 12 15 17 10"/>
              <line x1="12" y1="15" x2="12" y2="3"/>
            </svg>
            <span>下载</span>
          </a>
        </div>
      </div>
      <div class="final-media">
        <img class="final-image" alt="最终上色结果" src="${url}">
      </div>
    `;
    card.querySelector(".final-path").textContent = imagePath;
    card.querySelector(".final-path").title = imagePath;
    card.querySelector(".final-image").addEventListener("click", () => openLightbox(url));
    card.querySelector('[data-action="compare"]').addEventListener("click", (e) => {
      e.stopPropagation();
      toggleCompareView(card, imagePath);
    });
    chat.appendChild(card);
    scrollToBottom();
  }

  // ── 对比视图 (swipe slider) ────────────────────────────────────
  function toggleCompareView(card, afterPath) {
    if (!originalImagePath) {
      alert("找不到原图路径，无法对比");
      return;
    }
    const media = card.querySelector(".final-media");
    const btn = card.querySelector('[data-action="compare"]');
    const isCompare = card.classList.toggle("compare-mode");

    if (!isCompare) {
      media.innerHTML = `<img class="final-image" alt="最终上色结果" src="/image?path=${encodeURIComponent(afterPath)}">`;
      media.querySelector(".final-image").addEventListener("click", () =>
        openLightbox(`/image?path=${encodeURIComponent(afterPath)}`)
      );
      btn.classList.remove("active");
      btn.querySelector("span").textContent = "对比原图";
      return;
    }

    const beforeUrl = `/image?path=${encodeURIComponent(originalImagePath)}`;
    const afterUrl = `/image?path=${encodeURIComponent(afterPath)}`;
    media.innerHTML = `
      <div class="compare-view">
        <img class="compare-base" src="${beforeUrl}" alt="原图">
        <img class="compare-after" src="${afterUrl}" alt="上色后">
        <span class="compare-tag compare-tag-before">原图</span>
        <span class="compare-tag compare-tag-after">上色后</span>
        <div class="compare-handle"></div>
      </div>
    `;
    btn.classList.add("active");
    btn.querySelector("span").textContent = "关闭对比";
    bindCompareDrag(media.querySelector(".compare-view"));
  }

  function bindCompareDrag(view) {
    const handle = view.querySelector(".compare-handle");
    const after = view.querySelector(".compare-after");
    let dragging = false;

    function update(clientX) {
      const rect = view.getBoundingClientRect();
      let pct = ((clientX - rect.left) / rect.width) * 100;
      pct = Math.max(0, Math.min(100, pct));
      handle.style.left = pct + "%";
      after.style.clipPath = `inset(0 0 0 ${pct}%)`;
    }

    const onDown = (e) => {
      dragging = true;
      document.body.style.cursor = "ew-resize";
      update(e.touches ? e.touches[0].clientX : e.clientX);
      e.preventDefault();
    };
    const onMove = (e) => {
      if (!dragging) return;
      update(e.touches ? e.touches[0].clientX : e.clientX);
    };
    const onUp = () => {
      dragging = false;
      document.body.style.cursor = "";
    };

    handle.addEventListener("mousedown", onDown);
    handle.addEventListener("touchstart", onDown, { passive: false });
    view.addEventListener("mousedown", (e) => {
      if (e.target === handle) return;
      onDown(e);
    });
    window.addEventListener("mousemove", onMove);
    window.addEventListener("touchmove", onMove, { passive: false });
    window.addEventListener("mouseup", onUp);
    window.addEventListener("touchend", onUp);
  }

  // ─── Send ───────────────────────────────────────────────────────
  function send() {
    const text = input.value.trim();
    if (!text && !pendingImagePath) return;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      appendError("未连接到服务器，请等待重连...");
      return;
    }

    if (pendingImagePath) {
      originalImagePath = pendingImagePath;
    }
    appendUserMessage(text, pendingImagePath);
    ws.send(
      JSON.stringify({
        content: text,
        image_path: pendingImagePath,
      })
    );

    input.value = "";
    autoResize();
    clearUploadPreview();
    setStatus("busy", "LLM 思考中...");
    resetProgress();
    startTyping();
    updateSendButton();
    scrollToBottom();
  }

  function updateSendButton() {
    btnSend.disabled = !(input.value.trim() || pendingImagePath);
  }

  // ─── Upload ─────────────────────────────────────────────────────
  async function uploadFile(file) {
    if (!file || !file.type.startsWith("image/")) return;

    const fd = new FormData();
    fd.append("file", file, file.name || "image.png");

    try {
      setStatus("busy", "上传中...");
      const res = await fetch("/upload", { method: "POST", body: fd });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      pendingImagePath = data.path;
      previewImg.src = data.url;
      uploadPath.textContent = data.path;
      uploadPath.title = data.path;
      uploadPreview.hidden = false;
      setStatus("ready", "已就绪");
      updateSendButton();
      input.focus();
    } catch (err) {
      setStatus("error", "上传失败");
      appendError(`上传失败: ${err.message}`);
    }
  }

  function clearUploadPreview() {
    pendingImagePath = null;
    uploadPreview.hidden = true;
    previewImg.src = "";
    uploadPath.textContent = "";
    fileInput.value = "";
    updateSendButton();
  }

  // ─── Lightbox ──────────────────────────────────────────────────
  function openLightbox(src) {
    lightboxImg.src = src;
    lightbox.hidden = false;
  }
  function closeLightbox() {
    lightbox.hidden = true;
    lightboxImg.src = "";
  }

  // ─── Helpers ───────────────────────────────────────────────────
  function escapeHtml(s) {
    return String(s).replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        }[c])
    );
  }

  function scrollToBottom() {
    requestAnimationFrame(() => {
      chat.scrollTop = chat.scrollHeight;
    });
  }

  function autoResize() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 180) + "px";
  }

  // ─── Event wiring ──────────────────────────────────────────────
  btnSend.addEventListener("click", send);

  input.addEventListener("input", () => {
    autoResize();
    updateSendButton();
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      send();
    }
  });

  btnUpload.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) uploadFile(file);
  });
  clearUpload.addEventListener("click", clearUploadPreview);

  function bindChipButtons() {
    $$(".chip").forEach((btn) => {
      if (btn.dataset.bound) return;
      btn.dataset.bound = "1";
      btn.addEventListener("click", () => {
        if (btn.dataset.action === "example") {
          useExampleImage();
          return;
        }
        input.value = btn.dataset.prompt;
        autoResize();
        updateSendButton();
        input.focus();
      });
    });
  }
  bindChipButtons();

  async function useExampleImage() {
    try {
      setStatus("busy", "加载示例图...");
      const res = await fetch("/example-upload", { method: "POST" });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      pendingImagePath = data.path;
      previewImg.src = data.url;
      uploadPath.textContent = data.path;
      uploadPath.title = data.path;
      uploadPreview.hidden = false;
      input.value = "帮我上色";
      autoResize();
      updateSendButton();
      setStatus("ready", "已就绪");
      // 自动发送
      send();
    } catch (err) {
      setStatus("error", "示例图加载失败");
      appendError(`示例图加载失败: ${err.message}`);
    }
  }

  // ── 参数侧栏 ─────────────────────────────────────────────────────
  const paramSliders = [
    { key: "guidance_scale",      id: "ps-guidance",   display: "pv-guidance",   decimals: 1 },
    { key: "num_inference_steps", id: "ps-steps",      display: "pv-steps",      decimals: 0 },
    { key: "controlnet_scale",    id: "ps-controlnet", display: "pv-controlnet", decimals: 2 },
  ];

  function updateParamDisplay(key) {
    const cfg = paramSliders.find((p) => p.key === key);
    if (!cfg) return;
    const slider = $("#" + cfg.id);
    const display = $("#" + cfg.display);
    if (slider && display) {
      display.textContent = parseFloat(slider.value).toFixed(cfg.decimals);
      currentParams[key] = parseFloat(slider.value);
    }
  }

  function updateDrawerStatus() {
    const ready = currentTaskId && finalImagePath;
    btnApply.disabled = !ready;
    if (!currentTaskId) {
      drawerStatus.className = "drawer-status warning";
      drawerStatus.textContent = "还没有任务。先上传图并完成一次上色，才能重跑。";
    } else if (!finalImagePath) {
      drawerStatus.className = "drawer-status warning";
      drawerStatus.textContent = "任务已创建但还未完成后处理。";
    } else {
      drawerStatus.className = "drawer-status";
      drawerStatus.textContent = `就绪 · 任务 ${currentTaskId}`;
    }
  }

  function applyPreset(name) {
    const preset = PARAM_PRESETS[name];
    if (!preset) return;
    Object.assign(currentParams, preset);
    paramSliders.forEach((cfg) => {
      const slider = $("#" + cfg.id);
      slider.value = preset[cfg.key];
      updateParamDisplay(cfg.key);
    });
    $("#ps-seed").value = preset.seed;
    $("#pv-seed").textContent = preset.seed === -1 ? "随机" : preset.seed;
    currentParams.seed = preset.seed;
  }

  function applyParamsAndRerun() {
    if (!currentTaskId || !ws || ws.readyState !== WebSocket.OPEN) return;

    const p = currentParams;
    const seedText = p.seed === -1 ? "-1（随机）" : String(p.seed);
    const content =
      `请按以下参数重跑（任务 ${currentTaskId}）：\n` +
      `- guidance_scale = ${p.guidance_scale}\n` +
      `- num_inference_steps = ${p.num_inference_steps}\n` +
      `- controlnet_scale = ${p.controlnet_scale}\n` +
      `- seed = ${seedText}\n\n` +
      `请调用 update_inference_params 更新参数，用 reset_to_step 回退到 inference，` +
      `然后 run_inference + postprocess，最后 get_task_result。`;

    appendUserMessage(content, null);
    ws.send(JSON.stringify({ content }));
    setStatus("busy", "LLM 思考中...");
    resetProgress();
    startTyping();
    closeDrawer();
  }

  function openDrawer() {
    drawer.hidden = false;
    drawerBackdrop.hidden = false;
    updateDrawerStatus();
  }
  function closeDrawer() {
    drawer.hidden = true;
    drawerBackdrop.hidden = true;
  }

  // 滑块绑定
  paramSliders.forEach((cfg) => {
    const slider = $("#" + cfg.id);
    if (!slider) return;
    slider.addEventListener("input", () => updateParamDisplay(cfg.key));
    updateParamDisplay(cfg.key);
  });
  $("#ps-seed").addEventListener("input", (e) => {
    const v = parseInt(e.target.value, 10);
    if (!Number.isNaN(v)) {
      currentParams.seed = v;
      $("#pv-seed").textContent = v === -1 ? "随机" : v;
    }
  });
  btnRandomSeed.addEventListener("click", () => {
    const s = Math.floor(Math.random() * 2_147_483_647);
    $("#ps-seed").value = s;
    $("#pv-seed").textContent = s;
    currentParams.seed = s;
  });
  $$(".preset-btn").forEach((btn) => {
    btn.addEventListener("click", () => applyPreset(btn.dataset.preset));
  });
  btnDrawer.addEventListener("click", openDrawer);
  btnDrawerClose.addEventListener("click", closeDrawer);
  drawerBackdrop.addEventListener("click", () => {
    closeDrawer();
    closeLLMDrawer();
  });
  btnApply.addEventListener("click", applyParamsAndRerun);

  // LLM 配置抽屉
  btnLLM.addEventListener("click", openLLMDrawer);
  btnLLMClose.addEventListener("click", closeLLMDrawer);
  btnLLMApply.addEventListener("click", applyLLMConfig);
  $$(".provider-tab").forEach((tab) => {
    tab.addEventListener("click", () => switchProviderTab(tab.dataset.provider));
  });

  btnClear.addEventListener("click", () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      resetChatDOM();
      return;
    }
    if (!chat.querySelector(".message, .tool-card, .final-card")) {
      return; // 已是空状态
    }
    if (!confirm("清空当前对话？服务端会话历史会同步重置。")) return;
    ws.send(JSON.stringify({ type: "reset" }));
  });

  // Drag & drop
  let dragCounter = 0;
  window.addEventListener("dragenter", (e) => {
    if (e.dataTransfer && Array.from(e.dataTransfer.types).includes("Files")) {
      dragCounter++;
      dropzone.hidden = false;
    }
  });
  window.addEventListener("dragleave", () => {
    dragCounter--;
    if (dragCounter <= 0) {
      dragCounter = 0;
      dropzone.hidden = true;
    }
  });
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => {
    e.preventDefault();
    dragCounter = 0;
    dropzone.hidden = true;
    const file = e.dataTransfer && e.dataTransfer.files[0];
    if (file) uploadFile(file);
  });

  // Paste image
  window.addEventListener("paste", (e) => {
    if (!e.clipboardData) return;
    const item = Array.from(e.clipboardData.items).find((i) =>
      i.type.startsWith("image/")
    );
    if (item) uploadFile(item.getAsFile());
  });

  // Lightbox close
  lightbox.addEventListener("click", closeLightbox);
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !lightbox.hidden) closeLightbox();
  });

  // ─── Init ──────────────────────────────────────────────────────
  loadServerConfig();
  connect();
  updateSendButton();
  autoResize();
})();
