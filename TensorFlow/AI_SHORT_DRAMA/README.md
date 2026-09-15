# AI_SHORT_DRAMA

一个端到端的 AI 竖屏短剧自动化生产平台：一句创意 -> 故事引擎（8 个 LLM Agent）-> 质检官/总编审
-> 人工审核 -> 镜头生成（三级自动重试 + QC）-> ffmpeg 成片 -> 剧集 manifest -> 多平台发布 ->
数据回流分析。项目最初由 4 个并行开发的 agent 搭出各模块骨架，随后经过接口整合、免费开源模型
接入，最后一轮（v0.2）做了生产化改造：端到端编排、数据库打通、可靠性、可观测性、鉴权、测试与 CI。
这份 README 记录了平台现状、怎么跑、以及仍然明确不能做到的事。

**快速开始（本机，不需要 Postgres/Redis/GPU/API key）：**

```bash
pip install -r requirements.txt -r 13_INFRA/requirements-infra.txt
make test          # 53 个用例：mock LLM + dummy 生成器 + SQLite + eager 队列 + 真实 ffmpeg
make api           # http://localhost:8000/docs，X-API-Key: dev-key
curl -X POST localhost:8000/projects -H 'X-API-Key: dev-key' -H 'Content-Type: application/json' -d '{"name":"demo"}'
curl -X POST localhost:8000/pipelines/episodes -H 'X-API-Key: dev-key' -H 'Content-Type: application/json' \
  -d '{"project_id":"<上一步返回的 id>","idea":"豪门千金重生复仇，携手冷峻总裁夺回家族企业","episode_numbers":[1,2]}'
# 返回 run_id；GET /pipelines/{run_id} 看到 awaiting_review 后 POST /pipelines/{run_id}/approve 进入生产
```

## 平台总览（v0.2 生产化改造）

**一次流水线运行（PipelineRun）经过的阶段：**

```
POST /pipelines/episodes
   |
   v  story_task（llm 队列）
02 故事引擎 LangGraph：StoryAgent -> 章节规划师(ChapterPlannerAgent) -> CharacterAgent
   -> [每集] EpisodeAgent(带前情提要) -> ScreenplayAgent -> StoryboardAgent
   -> 质检官(QCOfficerAgent: 代码硬校验 + LLM 语义检查)
   -> 总编审(ChiefEditorAgent: approve / revise / reject)
        revise -> ScreenplayAgent 按 required_changes 重写 -> 重新分镜 -> 再审（最多 N 轮）
   -> 交叉校验（角色 id / 场次 / 镜头引用闭合）-> PromptAgent 或 jinja 模板出图像/视频 prompt
   |  全部对象落库（03 schema 原样存 JSON 列，主键带项目前缀）；每次 LLM 调用记 token/成本
   v  pipeline_gate_task
REQUIRE_HUMAN_REVIEW=true -> run.status=awaiting_review，等 POST /pipelines/{run_id}/approve
   |
   v  start_production：chord([shot_task x N] -> render_task) | manifest_task | publish_task
每个镜头（image 队列）：解析角色参考图/LoRA -> 生成 -> QC（CLIP 相似度）
   -> 不过就走三级重试阶梯：同参数换 seed -> AI 改写提示词 -> 降分辨率(1080p->720p->540p)
   -> 每次尝试都留 Asset + QCReport；阶梯用尽标 failed 但不拖垮其它镜头
渲染（qc 队列）：09_POST 把通过的关键帧/视频拼成竖屏成片 + SRT 字幕 -> Episode.mp4
manifest：10_EPISODES/episode_manifest.json（含 QC 报告引用、各平台发布状态）
发布：11_PUBLISH（PUBLISH_DRY_RUN=true 只校验不发）-> publish_status_task 轮询平台侧状态回写
```

**主要接口**（全部需要 `X-API-Key`；`/health`、`/metrics` 除外）：

| 接口 | 作用 |
|---|---|
| `POST /pipelines/episodes` | 一句创意起一次运行（参数：集数、场次数、prompt_mode、planner、是否 AI 编审……） |
| `GET /pipelines/{run_id}` | 阶段、编审记录、失败镜头、成片/manifest 路径、错误 |
| `POST /pipelines/{run_id}/approve` / `reject` / `rerender` | 人工审核放行 / 打回 / 补镜头后重渲染 |
| `GET /pipelines/{run_id}/usage` | 这次运行的 LLM 调用数、token、估算成本（按 agent 分组） |
| `GET /episodes/{id}/script`、`POST /episodes/{id}/review` | 审核页面用：剧本 + 质检官/总编审判定；单集放行 |
| `POST /tasks`（queue=llm/image/shot/video/tts/lipsync/qc/analytics） | 单个任务入队，payload 按队列 Pydantic 校验；`GET /tasks/schemas` 给 JSON Schema |
| CRUD：`/projects` `/characters` `/episodes` `/shots` `/assets` | 带 `limit/offset` 分页，`X-Total-Count` 响应头 |

**配置**：全部字段在 [`13_INFRA/config.py`](13_INFRA/config.py)（pydantic-settings），示例见
[`.env.example`](.env.example)。任何字段可用 `<NAME>_FILE=/run/secrets/...` 从文件读取。
`API_KEYS` 为空时受保护接口返回 503 而不是放行。

**部署**：[`13_INFRA/docker/docker-compose.yml`](13_INFRA/docker/docker-compose.yml)
= Postgres + Redis + migrate（Alembic）+ api + 六个队列 worker + Prometheus + Grafana。
Dockerfile 现在拷贝全部模块（之前只拷了 3 个目录，worker 起来就 ImportError），
`INSTALL_ML=1` 构建参数决定是否装 torch/diffusers。

**可靠性**：所有任务继承 [`BaseTask`](13_INFRA/queue/base_task.py)——瞬时错误
（网络/429/5xx/`TransientLLMError`/`TransientProviderError`）指数退避自动重试、`acks_late`、
worker 被杀消息回队；API provider 的 HTTP 调用统一走带退避的
[`request_with_retry`](07_GENERATION/http_retry.py)；会话历史读-改-写加分布式锁
（Redis Lock / fcntl）；数据库主键带项目前缀，两个项目的 `ep_001` 不会互相覆盖；
重跑同一集先清旧分镜（幂等）。

**可观测性**：结构化 JSON 日志（request_id / task_id 贯穿），Prometheus 指标
（任务耗时/成败、重试阶梯触发、QC 判定、LLM token 与成本、流水线状态），
`/metrics` + worker 9100 端口，Grafana 面板在 `13_INFRA/monitoring/`。

**测试与 CI**：`tests/` 下 5 个文件 + 原有 2 个，共 53 个用例（`make test`），
覆盖端到端流水线、三级重试阶梯、编审回灌、会话锁并发、发布状态轮询、分析接入、
ffmpeg 组装、训练数据集构造；`ruff` 全项目通过；GitHub Actions 见仓库根
`.github/workflows/ai_short_drama.yml`（lint + 测试 + Alembic 从零迁移 + Docker 构建）。

**如果你想自己动手重写一遍（不是直接用这份参考实现）**，先看
[`ARCHITECTURE.md`](ARCHITECTURE.md)（为什么要这样分层、核心设计决策）和
[`BUILD_ORDER.md`](BUILD_ORDER.md)（建议的分阶段实现顺序 + 每阶段验收标准）。

## 目录结构与数据流向

```
01_CONTENT          故事圣经 / 角色小传 / 剧本，人工手写的 Markdown
        |
        v
02_STORY_ENGINE     LLM Agent 把 01_CONTENT 转成结构化记录
        |            (StoryBible -> SeasonArc -> Characters -> Episode -> Script ->
        |             Scene -> Shot -> DialogueLine -> ImagePrompt/VideoPrompt)
        v
03_STRUCTURED_DATA  schemas.py：上述结构的权威 Pydantic 模型，外加 02 的 demo 产出的
        |           一份完整示例 *.json 文件
        v
04_DATASET          把参考图片/音频转成 datasets.Dataset manifest
        |           (metadata/ 下的 character/scene/style/voice/video *.jsonl)
        v
05_TRAINING         消费 04_DATASET 数据集的 LoRA / 微调脚本
        |           (character_lora, style_lora, voice, video, llm)
        v
06_MODELS           一层很薄的模型注册表/配置层，记录每个领域用哪个模型 checkpoint
        |           (base 模型 + 训练出的 LoRA，覆盖 asr/image/lipsync/llm/tts/video/vlm)
        v
07_GENERATION       按 Shot 真正调用生成后端：image、video、tts、lipsync
        |           （真实的 provider 类封装，见下方"哪些是真实实现"）
        v
08_QC               给每个生成的素材打分（CLIP 做角色一致性、场景对齐、视频/音频探测）
        |           并决定通过还是重试
        v
09_POST             基于 ffmpeg 的后期：拼接、字幕、配乐/音效混音、渲染
        v
10_EPISODES         每集最终的 manifest（EP001/、EP002/），引用 shot/asset
        v
11_PUBLISH          各平台发布客户端（youtube、tiktok、reelshort、dramabox、goodshort）
        v
12_ANALYTICS        对已发布剧集做 CTR / 留存 / 收入 / 实验分析

13_INFRA            横切关注点：Postgres 模型 + FastAPI CRUD、Celery 任务队列
                    （分发到 07_GENERATION/08_QC）、Docker/监控/存储。
                    这是唯一合理地需要触达其他所有顶层模块的模块。
```

Shot 是贯穿整条流水线的关联键：`03_STRUCTURED_DATA.Shot.id` -> `ImagePrompt.shot_id` /
`VideoPrompt.shot_id` -> `07_GENERATION` 里的 asset 记录（`shot_id` 字段）-> QC 报告
（`shot_id`）-> `10_EPISODES` 的 manifest（`shot_ids`）。

## 哪些是真实可跑的实现，哪些是占位

**真实可跑、完全离线（不需要密钥、GPU、联网）：**
- `02_STORY_ENGINE` demo：全程用 `MockLLMProvider` + 手写 fixture 离线跑通；
  `AnthropicProvider`/`OpenAIProvider` 是真实的 API 封装，但要调用真实 LLM 需要
  `ANTHROPIC_API_KEY`/`OPENAI_API_KEY`。
- `03_STRUCTURED_DATA/schemas.py` —— 权威 Pydantic 模型。把它当作唯一真源，其他地方
  不应该重复定义 `Shot`、`Character` 等。
- `04_DATASET` —— manifest 指向的是 PIL 生成的占位 PNG
  （`04_DATASET/images/generate_placeholder_images.py`），不是真实拍摄的素材；
  voice/video manifest 里的行引用的是磁盘上并不存在的路径（按脚本自己的 docstring 说明，
  这些行的作用只是用来跑通 loader 代码）。
- `07_GENERATION.image.image_generator.DummyImageGenerator` —— 画一张带文字标注的
  占位 PNG；`07_GENERATION/demo.py` 和 `13_INFRA` 的 `image_task` 都用它，这样整条
  素材流水线（生成 -> 保存 -> AssetRecord -> 数据库 -> QC）可以在没有模型权重的情况下
  冒烟测试。
- `08_QC.character.consistency_checker` 用的是真实 CLIP（`openai/clip-vit-base-patch32`，
  已缓存在本机 `~/.cache/huggingface` 下）—— 这个是真跑推理，不是 mock。
- `09_POST` —— 通过 subprocess 调真实的 `ffmpeg`/`ffprobe`；需要这两个二进制在 `PATH` 里。
- `12_ANALYTICS` —— 对合成示例数据做真实的 pandas/scipy 计算。
- `13_INFRA/api` 和 `13_INFRA/queue` —— 真实的 FastAPI + SQLAlchemy + Celery 接线；
  跑在 SQLite + Celery eager 模式下，不需要任何外部服务（见 `13_INFRA/api/demo.py`）。

**真实的 API 封装，但这台机器没有对应的密钥：**
- `07_GENERATION.video.video_generator`（`RunwayVideoProvider`、`PikaVideoProvider`）、
  `07_GENERATION.audio.tts_generator.ElevenLabsTTSProvider`、
  `07_GENERATION.voice.voice_cloner.ElevenLabsVoiceCloneProvider`、
  `07_GENERATION.lipsync.lipsync_generator.SyncSoLipSyncProvider` —— 这些都是设计成
  在构造时如果对应的 `*_API_KEY` 环境变量没设置，就抛 `NotConfiguredError`，不会静默
  失效。这几类现在都有一个免费、不需要 API key 的本地替代方案（见下一节）。

**真实、本地、免费（不需要 API key），但这台机器没有跑满血默认模型所需的 GPU/显存：**
- `07_GENERATION.image.image_generator.DiffusersImageGenerator`（需要 `torch` + GPU
  以及下载好的扩散模型权重）。
- `02_STORY_ENGINE.agents.base.LocalTransformersProvider`、
  `07_GENERATION.video.video_generator.MochiLocalVideoProvider`、
  `07_GENERATION.audio.tts_generator.BarkLocalTTSProvider`、
  `07_GENERATION.audio.asr_transcriber.WhisperLocalASRProvider` —— 见下方详情，
  除非特别标注，都已经在本机用小/快的 checkpoint **真实跑通过一次完整流程**；生产环境
  把 `model_id` 换成各个类 docstring 里列出的默认大模型即可。
- `13_INFRA/queue` 里的 `image_task`/`video_task`/`tts_task`/`lipsync_task` 现在都调用
  这些真实的类（见下方"修复记录"）；没有 API key 时，video/tts/lipsync 这三个任务会
  正确地抛 `NotConfiguredError`，image 任务会用 `DummyImageGenerator` 成功跑通。

**只有结构骨架，没有真实后端，也不假装自己有：**
- `11_PUBLISH/reelshort`、`11_PUBLISH/dramabox`、`11_PUBLISH/goodshort` —— 这三家
  平台没有公开的开发者 API。它们的客户端是一个可配置的 `GenericHTTPAdapter` 骨架
  （见各自的 `config.example.yaml`），通过 `--dry-run` 校验 payload 结构；不是真实对接，
  在 `config.yaml` 指向一个真实存在的 endpoint 之前都会报错。`youtube` 和 `tiktok`
  封装的是真实的公开 API（YouTube Data API、TikTok Content Posting API）。
- 本仓库没有任何代码去抓取真实的 ReelShort/DramaBox/GoodShort 数据，也没有下载任何
  受版权保护的训练数据 —— 按项目约束，不要新增这类代码。

## 免费开源模型选项（不需要 API key）

每一类生成任务现在都至少有一个**完全本地、免费、不需要 API key**的选项——只需要 GPU
（或者有耐心用 CPU 跑）和存权重的磁盘空间。下表里每个 provider 类都被真实实例化过，
除非特别标注"未实跑"，都在本机（CPU，用小/快的 checkpoint）跑通过一次完整流程来确认
代码逻辑是对的，而不只是读代码觉得对。

| 类别 | Provider 类 | 免费模型（默认值） | 许可证 | 验证情况 |
|---|---|---|---|---|
| LLM | `02_STORY_ENGINE/agents/base.py::LocalTransformersProvider` | `microsoft/Phi-3.5-mini-instruct` | MIT | 通过 `13_INFRA/queue/llm_task.py` 用更小的测试 checkpoint（`HuggingFaceTB/SmolLM2-360M-Instruct`，同样 MIT）真实跑通——回答正确，返回结构符合约定 |
| 图片 | `07_GENERATION/image/image_generator.py::DiffusersImageGenerator` | `stabilityai/stable-diffusion-xl-base-1.0` / `black-forest-labs/FLUX.1-schnell` | OpenRAIL++-M / Apache-2.0 | 结构上已核实（真实的 diffusers API，这次没有重新实跑；`07_GENERATION/demo.py` 为了保持快速，用的是 `DummyImageGenerator` 走通流程） |
| 视频 | `07_GENERATION/video/video_generator.py::MochiLocalVideoProvider` | `genmo/mochi-1-preview` | Apache-2.0 | 对照本机已安装的 `diffusers==0.40.0`（`MochiPipeline` 签名、`export_to_video`）核实过接口正确；没有端到端实跑——这个 checkpoint 有几十 GB，需要大显存 GPU |
| TTS | `07_GENERATION/audio/tts_generator.py::BarkLocalTTSProvider` | `suno/bark-small` | MIT | **在 CPU 上真实跑通**——生成了一个真实的 357KB wav 文件 |
| 声音克隆 | `07_GENERATION/voice/voice_cloner.py::CoquiXTTSLocalVoiceCloneProvider` + `tts_generator.py::CoquiXTTSLocalTTSProvider` | `tts_models/multilingual/multi-dataset/xtts_v2` | Coqui Public Model License 1.0.0 —— **默认非商用** | 在项目主环境（`transformers` 5.5.0）下 import 就报错，但在单独的 `.venv_xtts` 虚拟环境（`transformers<5.0` + 补装 `torch`/`torchaudio`/`coqui-tts[codec]`）里**真实跑通**——真的克隆了一段参考音频、生成了一个 5.2 秒的 wav，用 `ffprobe` 核实过是合法音频文件。踩了两个坑：torchcodec 在 macOS 下需要 `DYLD_FALLBACK_LIBRARY_PATH` 指到 Homebrew ffmpeg 的 lib 目录才能 dlopen 成功；首次下载 XTTS-v2 权重会弹出交互式 CPML 非商用协议确认，`COQUI_TOS_AGREED=1` 可以跳过，但这是用户本人确认同意协议的动作，这次是征得你同意才设置的，仓库代码本身不会替你设置 |
| ASR（用于字幕） | `07_GENERATION/audio/asr_transcriber.py::WhisperLocalASRProvider` | `openai/whisper-large-v3` | Apache-2.0/MIT | **在 CPU 上真实跑通**（测试用的是 `openai/whisper-tiny`），拿上面 Bark 生成的 wav 反过来做转写——转写结果正确，而且它输出的 `{"segments": [...]}` 不改一个字段就能直接喂给 `09_POST/subtitle/subtitle.py::cues_from_whisper_result()`（已验证） |
| QC（图文相似度） | `08_QC/character/consistency_checker.py`、`08_QC/scene/scene_qc.py` | `openai/clip-vit-base-patch32` | MIT | 早前一轮已经是真实 CLIP 推理（见上文） |
| 对口型 | —— | —— | —— | **没有实现 provider。** 见 `06_MODELS/lipsync/registry.json`：目前知名的开源对口型模型要么许可证不清楚（MuseTalk、SadTalker），要么明确只能用于研究/非商用（Wav2Lip）。在正式选型前需要重新调研。 |

`13_INFRA/queue/llm_task.py` 现在默认使用 `LocalTransformersProvider`（免费）——只要
`payload["model"]` 不是 `claude*`/`gpt*` 这种名字，队列就会跑在免费的开源模型上，除非
你显式要求用付费 API 模型。

## LLM 会话记忆（历史消息长期保留，触顶才裁剪）

所有 LLM 调用（六个 agent + `llm_task` 队列任务）都支持带上同一会话的历史消息，
实现在 `02_STORY_ENGINE/agents/memory.py`：

- **持久化**：`ConversationStore` 抽象，三种实现——`InMemoryConversationStore`（测试）、
  `FileConversationStore`（每个会话一个 JSON 文件，原子写入）、`RedisConversationStore`
  （多 worker 共享，可选 TTL，不设即永不过期）。存储里的完整历史**永远不主动删**。
- **窗口**：`ConversationMemory.window(system_prompt, user_prompt)` 按
  `模型上下文上限 - 输出余量 - system prompt - 本次 prompt` 算预算，没超就把历史原样全带；
  超了才从最旧的轮次开始成对丢（`compaction="truncate"`，默认），或先用同一个 LLM 压成
  摘要再接着带（`compaction="summarize"`）。
- **上下文上限**：`context_window_for(model)` 按模型名前缀查表（claude 200k、gpt-4o 128k、
  Phi-3.5 128k……未知模型保守按 8k）；`LocalTransformersProvider` 加载权重后会用模型 config
  里的 `max_position_embeddings` 覆盖表里的估计值。
- **只记成功轮次**：`BaseAgent.generate()` 里 JSON 校验失败的重试不进历史，避免历史被
  报错信息灌满。

用法：

```python
from agents import StoryAgent, CharacterAgent, FileConversationStore, build_memory

memory = build_memory(provider, context_id="project_001", store=FileConversationStore("contexts"))
story_agent = StoryAgent(provider, memory=memory)
character_agent = CharacterAgent(provider, memory=memory)  # 共用同一份历史
```

`02_STORY_ENGINE/demo.py` 已经这样接好：历史落在 `02_STORY_ENGINE/demo_output/conversations/`，
重跑 demo 会先加载上次的轮次再继续。队列侧 `llm_task` 的 payload 传 `context_id` 就会续接
该会话（返回值多了 `context_id` 和 `history_turns`），存储后端由环境变量决定：
`LLM_CONTEXT_REDIS_URL`（生产，docker-compose 已给 `worker_llm` 配好 `redis://redis:6379/1`）
或 `LLM_CONTEXT_DIR`（本地文件，默认系统临时目录下 `ai_short_drama_contexts`）。
测试：`pytest 02_STORY_ENGINE/tests/test_memory.py`（14 个用例，不需要 API key 和模型权重；
Redis 存储另外用本机临时起的 `redis-server` 实跑验证过读写、TTL 和经 `llm_task` 续接）。

每个类别更完整的免费/开源候选列表，都带许可证说明（拿不准的地方明确写"需要自行核实"，
不瞎猜）：`06_MODELS/{llm,image,video,vlm,tts,asr,lipsync}/registry.json`。

**关于模型来源的提醒：** 有几个最强的免费开源模型是中国出品的（LLM/VLM 里的 Qwen，
以及大多数靠谱的开源对口型模型）。它们之所以还留在 registry 里，是因为它们确实免费、
技术上确实强——但最初的项目需求里提到过倾向于排除中国出品的模型，尤其是视频这一层。
"选最强的"和"按来源排除"这两者之间的取舍，这里没有替你做决定，需要你按自己的实际
约束逐类别挑选，不能只看许可证。

## 对照《Agent 搭建指南》的落地清单（工具 / 记忆 / 多步骤 / 部署 / 监控 / 踩坑）

这个项目按"Agent 大脑 + 工具层 + 推理层 + 记忆 + 监控 + 人机协作"的标准结构落地，每一项都有对应代码和测试：

| 指南要求 | 本项目实现 | 代码位置 |
|---|---|---|
| 步骤 2 定义 Agent 核心行为（系统指令） | 每个 Agent 一份系统提示词文件，"指令是最好的训练"：枚举合法值直接写进 prompt | `02_STORY_ENGINE/prompts/*.txt`、`agents/base.py::_enum_cheatsheet` |
| 步骤 3 注册工具能力（`@agent.tool`） | `ToolRegistry.tool` 装饰器：函数签名自动生成参数 JSON Schema（pydantic），docstring 即工具说明 | `02_STORY_ENGINE/agents/tools.py` |
| 步骤 4 Memory（persistent, sqlite） | `ConversationMemory` + 四种存储：内存 / JSON 文件 / **SQLite** / Redis；历史永不删，触顶才裁剪或摘要压缩 | `agents/memory.py`（`SQLiteConversationStore`），`LLM_CONTEXT_SQLITE_PATH` |
| 步骤 5 多步骤工作流（思考 -> 工具 -> 思考 -> 回复） | `ToolAgent.run()`：模型每轮输出 `{"action": "tool"/"final"}`，工具结果回灌，直到给出最终答案；`DramaAssistantAgent` 是具体实例 | `agents/tool_agent.py`、`agents/drama_assistant.py` |
| 推理层：规划 -> 执行 -> 验证 -> 修正 -> 再执行 | 剧本层：质检官 + 总编审 + 回灌重写（LangGraph）；镜头层：三级重试阶梯；助理层：`run_rule_check` 验证后再决定是否标记 | `workflows/pipeline.py`、`07_GENERATION/retry_ladder.py` |
| Docker 部署 | 多阶段 Dockerfile + compose（Postgres / Redis / API / 每个队列一个 worker / Prometheus） | `13_INFRA/docker/` |
| 监控 1 Token 消耗 | 每次 LLM 调用 -> `llm_usage` 表 + `drama_llm_tokens_total` / `drama_llm_cost_usd_total` | `13_INFRA/queue/_common.py::make_usage_sink` |
| 监控 2 工具调用成功率 | 每次工具调用 -> `drama_tool_calls_total{tool,status}` + 耗时直方图 | `make_tool_call_sink`、`observability/metrics.py` |
| 监控 3 执行时间 | 队列任务耗时直方图 `drama_task_duration_seconds`，Agent 每次 run 的工具轮数直方图 | `queue/base_task.py`、`record_agent_run` |
| 监控 4 错误率 | `drama_task_total{status}` 的 failure 占比；降级次数 `drama_llm_fallback_total` | `GET /metrics`、`GET /metrics/summary`（四项汇总成 JSON） |
| 坑 1 无限循环 | `max_tool_rounds` 默认 **5**（`AGENT_MAX_TOOL_ROUNDS`）；连续相同 (工具, 参数) 被识别为原地打转并强制作答；仍不收敛抛 `AgentLoopError` | `tool_agent.py` |
| 坑 2 上下文爆炸 | 一次 run 的中间轮次不进长期记忆；历史按模型上下文上限裁剪，`compaction="summarize"` 用同一模型压成摘要 | `memory.py::ConversationMemory.window` |
| 坑 3 工具权限过大 | 工具级确认门：`requires_confirmation=True` 的工具默认拒绝；`AllowlistGate`（生产，`AGENT_TOOL_ALLOWLIST`）/ `ConsoleGate`（本地 y/n）/ `CallbackGate`（审批系统） | `tools.py`、`_common.py::build_confirmation_gate` |
| 坑 4 降级响应（多模型策略） | `FallbackProvider`：主模型限流/超时/没配 key 时按顺序切备用，带熔断冷却自动恢复；`LLM_FALLBACK_MODELS=gpt-4o-mini,microsoft/Phi-3.5-mini-instruct` | `agents/fallback_provider.py`、`provider_factory.build_provider` |
| 原则 人机协作 | 流水线级：`REQUIRE_HUMAN_REVIEW` 停在 `awaiting_review` 等 `/approve`；工具级：确认门 | `13_INFRA/orchestration.py` |

跑一遍看效果（全部 mock，不需要 API key）：

```bash
python3 02_STORY_ENGINE/demo.py          # 末尾 "Drama Assistant" 段：list_episodes -> run_rule_check -> 回复
pytest tests/test_tool_agent.py          # 20 个用例：工具/确认门/循环上限/重复检测/降级/SQLite 记忆/助理/四大指标
```

API 侧：`POST /pipelines/{run_id}/assistant {"question": "帮我检查第 1 集有没有引用错误"}` 把问题交给
`assistant_task`（llm 队列），返回答案、每一步工具调用记录、token 用量；`GET /metrics/summary` 看四大指标。

### Web 控制台（不用 Postman）

```bash
make api            # 本地起 API：SQLite + eager 队列 + mock LLM，不需要 Redis / API key
open http://localhost:8000/   # API key 填 dev-key
```

`13_INFRA/api/static/index.html` 是零构建、零依赖的单文件页面，由 FastAPI 在 `/`（或 `/ui`）直接托管，
Docker 镜像里同样自带。五个页面对应平台的全部能力：

| 页面 | 能做什么 |
|---|---|
| 流水线 | 填一句创意启动流水线；运行列表按状态筛；详情页看每集状态 / AI 编审结果 / 成片 / 发布记录；勾选集 **批准进入生产**、打回、重新渲染；看剧本正文与质检官/总编审判定；查本次 LLM 用量；运行中自动轮询 |
| 制片助理 | 对一次运行的产出提问（有快捷问题），聊天式展示 Agent 每一步调用了什么工具、参数、耗时、成功/被拒，以及 token 成本；生产模式自动轮询任务结果 |
| 项目 | 建项目、列项目、看项目下的角色和剧集 |
| 监控指标 | 四大指标卡片 + 按模型 / 工具 / 任务的明细，可 5 秒自动刷新，直达 Prometheus 原始数据 |
| 单步任务 | 选队列、填 payload（可一键填示例、看 JSON Schema）入队，查任务状态 |

API key 只存在浏览器 localStorage；页面本身不需要 key 就能打开，数据请求才带 `X-API-Key`。

## 跨模块 import 约定

每个顶层目录名都是数字开头的（`01_CONTENT`、`02_STORY_ENGINE`……），所以都不能作为
普通 `import` 语句的目标——`import 03_STRUCTURED_DATA` 会直接 `SyntaxError`，因为
`03_STRUCTURED_DATA` 不是合法的标识符。

**新写跨顶层模块的代码时（比如 `13_INFRA` 里要分发到其他模块的代码），统一用
`importlib.import_module` + 项目根目录开始的完整点号路径：**

```python
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]  # 按实际层级调整
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import importlib

image_generator = importlib.import_module("07_GENERATION.image.image_generator")
generator = image_generator.DummyImageGenerator()
```

不管中间目录有没有 `__init__.py` 这种写法都能用（Python 的命名空间包机制会兜底）；
而且和"把某个子目录塞进 sys.path 再裸 `from module import X`"这种写法不同，它不会跟
别的目录里同名的模块撞车，因为路径是完全限定的。这一点在本项目里很实际：仓库里有两个
`schemas.py`（`03_STRUCTURED_DATA/schemas.py` 和 `04_DATASET/metadata/schemas.py`，
定义的是完全不相关的东西——前者是故事/分镜的权威模型，后者是数据集 manifest 行的模型），
还有五个 `client.py`（每个 `11_PUBLISH` 平台一个）。`13_INFRA/queue/*.py` 和
`13_INFRA/database/alembic/env.py` 用的就是这个约定；已经直接验证过可以正常工作
（`importlib.import_module("13_INFRA.queue.llm_task")` 和
`importlib.import_module("03_STRUCTURED_DATA.schemas")` 在只把项目根目录加进
`sys.path` 的情况下都能成功）。

**现有代码里还能看到另外两种写法——保留原样，不是 bug：**
- `02_STORY_ENGINE`、`07_GENERATION` 自己的生成器模块、以及 `05_TRAINING`，是在文件
  开头把一两个特定的兄弟目录塞进 `sys.path`，然后用裸的 `import schemas as sch` /
  `from agents import X`。这样写没问题，因为这些文件每次都只需要一个强耦合的兄弟目录
  （比如 `02_STORY_ENGINE` 永远只需要 `03_STRUCTURED_DATA`），所以现在不存在真实的
  命名冲突。已验证可以正常工作（`02_STORY_ENGINE/demo.py`、`07_GENERATION/demo.py`
  都能干净地跑通）。如果以后要新增一个需要同时依赖两个、可能有同名模块的兄弟目录的
  文件，不要再沿用这种写法——改用上面的 `importlib.import_module` 约定。
- `11_PUBLISH/publish_cli.py` 用的是第三种写法，
  `importlib.util.spec_from_file_location` + `module_from_spec`，原因是它要把五个同名的
  `client.py` 都加载进同一个进程，需要在 `sys.modules` 里用不同的名字区分开。这正是上面
  `importlib.import_module` 约定天然能避免的那种冲突；保留原样是因为它已经能正常工作，
  重写也不会改变行为。

不要在没检查过的情况下，新增一个叫 `schemas.py`、`client.py`、`generator.py` 之类的文件——
先确认同一进程里可能出现在 `sys.path` 上的兄弟模块有没有已经用了这个文件名。

## Python 版本注意事项

这台机器默认的 `python3` 是 **3.14.2**。`datasets` 库（以及它用来做缓存 key 哈希的
`dill` 依赖）在 3.14 上是坏的：

```
TypeError: Pickler._batch_setitems() takes 2 positional arguments but 3 were given
```

这是一个真实的、可复现的报错（通过直接运行 `04_DATASET/dataset_loaders.py` 验证过）——
`dill` 的 `Pickler.save_module_dict` 重写调用了 `StockPickler.save_dict`，后者又调用
`self._batch_setitems(obj.items(), obj)`，而 Python 3.14 标准库 `pickle.Pickler
._batch_setitems` 的签名发生了变化，`dill` 还没跟上。这会影响所有 import `datasets`
并调用 `load_dataset`/`Dataset.from_list` 等方法的代码——具体来说就是
`04_DATASET/dataset_loaders.py` 以及 import 了它的 `05_TRAINING/*` 脚本。

这个问题不在本仓库里修（是上游 `dill`/`datasets` 的 bug，不是本仓库的 bug）。
**权宜之计：**给这部分代码单独建一个 Python 3.12 或 3.13 的虚拟环境，例如：

```
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt   # 或者只装 04_DATASET/05_TRAINING 那几段
```

仓库里其他部分（pydantic schema、FastAPI/Celery 基础设施、基于 ffmpeg 的后期、
数据分析、用 mock 或真实 LLM 的 Story Engine）在 3.14 上都能正常跑。

## 依赖

每个模块自己目录下都保留了 `requirements-*.txt`（`02_STORY_ENGINE` 是
`requirements.txt`）——原样未动，单独看某个模块时仍以它为准。项目根目录的
`requirements.txt` 是把它们汇总起来、按模块分段加了注释的聚合版，方便搭一个能跑
全部代码的环境。

## 如何跑通各模块的 demo

```
# 02 Story Engine：mock-LLM 流水线 -> 写出 03_STRUCTURED_DATA 格式的 JSON
python 02_STORY_ENGINE/demo.py

# 04 Dataset：生成示例 manifest（占位图片 + 编造的音频/视频行）
# 然后加载成 datasets.Dataset —— 需要 Python <=3.13，见上文
python 04_DATASET/metadata/generate_sample_manifests.py
python 04_DATASET/dataset_loaders.py

# 07 Generation：读取合成的 Shot+ImagePrompt 记录，生成占位图片，写出 AssetRecord
python 07_GENERATION/demo.py

# 08 QC：没有独立的 demo 脚本；直接 import 测试或调用 run_full_qc(...)，
# 例如 python -c "from importlib import import_module; ..." —— CharacterConsistencyChecker
# 需要本机已缓存的 CLIP 权重（~/.cache/huggingface 下已经有了）。

# 10 Episodes：校验一份剧集 manifest
python 10_EPISODES/episode_manifest.py 10_EPISODES/EP001/episode_manifest.json

# 11 Publish：对每个平台客户端的 payload 构建做 dry-run（不发真实网络请求）
python 11_PUBLISH/publish_cli.py --episode 10_EPISODES/EP001/episode_manifest.json --dry-run

# 12 Analytics：留存/CTR/收入/实验的示例报告
python 12_ANALYTICS/demo.py

# 13 Infra API + queue：FastAPI CRUD + Celery 任务往返，SQLite + 进程内
# Celery eager 模式，不需要 Postgres/Redis
python 13_INFRA/api/demo.py
# 或者：pytest 13_INFRA/api/test_demo.py

# 全平台端到端（创意 -> 编审 -> 审核 -> 镜头 -> 渲染 -> manifest -> 发布 dry-run）
pytest tests/test_pipeline_e2e.py

# 05 Training：从故事引擎产出构造 SFT 数据集，并评估一个模型的结构化输出可靠性
python 05_TRAINING/llm/build_sft_dataset.py --from-demo-output 02_STORY_ENGINE/demo_output --out data/sft.jsonl
python 05_TRAINING/llm/evaluate_structured_output.py --dataset data/sft.jsonl --provider mock --limit 20

# 本地起 API / worker（见 Makefile）
make api
make worker
```

## 整合 4 个 agent 的模块时修复的问题

这些模块是由互相看不到对方代码的 agent 并行开发的。以下接口不一致问题是通过读实际
代码（而不是轻信 agent 自己的总结）发现并修复的：

1. **重复的 Shot schema。** `04_DATASET/metadata/shot_schema.py` 是一份 agent 自己承认
   的占位 Shot/DialogueLine 模型（字段是 `shot_id`、`character_ids`、`environment_id`
   这种），是在 `03_STRUCTURED_DATA/schemas.py` 还不存在的时候写的。已删除；
   `07_GENERATION/demo.py`（唯一真正 import 过它的地方）现在直接从
   `03_STRUCTURED_DATA/schemas.py` import `Shot`、`ImagePrompt`、`CameraSpec` 等，
   用权威字段名构造（`id`、`episode`、`scene`、`shot`、`character`、`location`、
   `action`、`emotion`、`camera`、`duration`——注意 prompt 文本是存在单独的
   `ImagePrompt` 记录里，不在 `Shot` 本身上）。`04_DATASET/dataset_loaders.py` 和
   `04_DATASET/metadata/generate_sample_manifests.py` 实际上从来没有 import 过
   `Shot`/`shot_schema`（它们只用了不相关的 `04_DATASET/metadata/schemas.py` 里的
   manifest 行模型），所以这两个文件不需要改——删除后重新跑过一遍确认没问题。
2. **`13_INFRA/queue` 的任务 stub 假设了一个并不存在的
   `07_GENERATION.<domain>.generate(**payload)` 函数。** `image_task.py`、
   `video_task.py`、`tts_task.py`、`lipsync_task.py` 现在都调用真实的类：
   `DiffusersImageGenerator`/`DummyImageGenerator`（图片）、`VideoGenerator` +
   `RunwayVideoProvider`/`PikaVideoProvider`（视频）、`TTSGenerator` +
   `ElevenLabsTTSProvider`（tts）、`LipSyncGenerator` + `SyncSoLipSyncProvider`
   （对口型）。已在 Celery eager 模式下端到端验证：`image_task` 通过
   `DummyImageGenerator` 成功；另外三个正确地抛出 `NotConfiguredError`
   （缺 API key），而不是之前那种 `ModuleNotFoundError`/`NotImplementedError`。
   `qc_task.py` 不需要改——它本来就调用真实的
   `08_QC.reports.qc_report.run_full_qc`。`13_INFRA/api/demo.py` 里有一条测试断言
   是按*旧的、坏掉的*行为写的（断言 image 任务会以 `NotImplementedError` 失败）；
   已更新为断言现在正确的 `SUCCESS` 结果。
3. **跨目录 import 风格**——见上面的"跨模块 import 约定"。已验证
   `sys.path`-插入后裸 import 和 `importlib.import_module` 带点号路径这两种写法
   都能正常工作；`13_INFRA`（原本就部分在用）统一成 `importlib.import_module`，
   并把它写成新跨模块代码的约定，没有去改写其他地方约 20 个已经能正常工作的旧写法
   文件（今天没有真实冲突，见上面的注意事项）。
4. **Python 3.14 / `dill` 不兼容**——见上面"Python 版本注意事项"；通过实际运行代码
   确认是真实问题，没有修（是上游库的 bug），用虚拟环境的方式记录了权宜之计。

没有发现其他重复/不一致的模型定义：`13_INFRA/database/models.py` 里也定义了
`Character`/`Episode`/`Scene`/`Shot`，但那些是持久化层用的 SQLAlchemy ORM 表
（故意比领域 schema 简单——比如没有 `emotion`/`camera` 字段，因为数据库层不需要），
并且和 `13_INFRA/api/schemas.py` 是内部一致的；这是合理的关注点分离，不是命名冲突
bug。

## 接入免费开源模型时的修复记录

- `02_STORY_ENGINE/agents/base.py::LocalTransformersProvider`：用
  `transformers.pipeline("text-generation", ...)` 跑本地免费模型。
- `07_GENERATION/video/video_generator.py::MochiLocalVideoProvider`：用 diffusers 的
  `MochiPipeline` 跑 Genmo Mochi 1（Apache-2.0）。
- `07_GENERATION/audio/tts_generator.py::BarkLocalTTSProvider`：用 Suno Bark（MIT）
  做本地 TTS。**踩过一个真实的坑**：一开始把 voice 预设当成 `forward_params=
  {"history_prompt": voice_id}` 传给 pipeline，结果 Bark 内部的 `generate()` 期望
  `history_prompt` 已经是解析好的 dict（带 `semantic_prompt` 等 key），不是一个原始
  字符串，报 `TypeError: string indices must be integers, not 'str'`。看了
  `TextToAudioPipeline._sanitize_parameters`/`preprocess` 源码后发现正确做法是走
  `preprocess_params={"voice_preset": voice_id}`，让它转给 `BarkProcessor.__call__`
  的 `voice_preset` 参数。改完之后在 CPU 上真实生成了一个 357KB 的 wav，问题解决。
- `07_GENERATION/audio/asr_transcriber.py::WhisperLocalASRProvider`（新文件）：本地
  Whisper 转写，输出适配成 `09_POST/subtitle/subtitle.py::cues_from_whisper_result()`
  期望的 `{"segments": [{"start","end","text"}]}` 格式。用上面 Bark 生成的 wav 做了
  真实的 TTS -> ASR -> 字幕 cue 闭环测试，转写结果和原文一致。
- `07_GENERATION/voice/voice_cloner.py::CoquiXTTSLocalVoiceCloneProvider` +
  `07_GENERATION/audio/tts_generator.py::CoquiXTTSLocalTTSProvider`：Coqui XTTS-v2
  零样本声音克隆，免费但许可证默认非商用，已在 docstring 里显著标注。项目主环境
  （`transformers` 5.5.0）下装 `coqui-tts`（0.27.5）会在 import 阶段直接报
  `ImportError: cannot import name 'isin_mps_friendly' from
  transformers.pytorch_utils`——coqui-tts 的 XTTS 代码依赖一个 `transformers` 5.5.0
  里已经不存在的内部工具函数，是这两个库之间的版本冲突，不是这份 provider 代码本身
  的 bug。没有在主环境里降级 `transformers` 去修，因为已验证能跑的
  `LocalTransformersProvider`/`BarkLocalTTSProvider`/`WhisperLocalASRProvider`
  都依赖当前这个 5.5.0，贸然全局降级会有弄坏它们的风险。**后续在独立的
  `.venv_xtts` 虚拟环境里把它跑通了**：`pip install "transformers<5.0" coqui-tts
  pydantic && pip install torch torchaudio && pip install "coqui-tts[codec]"`。
  过程中又踩了两个真实的坑：① macOS 下 torchcodec 自带的原生库找不到 Homebrew
  ffmpeg 的动态库（`Library not loaded: @rpath/libavutil.57.dylib ... no
  LC_RPATH's found`），要显式设置
  `DYLD_FALLBACK_LIBRARY_PATH="$(brew --prefix ffmpeg)/lib"` 才能加载成功；
  ② 首次下载 XTTS-v2 权重会弹出交互式确认，要求同意 CPML 非商用协议条款，
  这一步征得你本人同意后才设置了 `COQUI_TOS_AGREED=1` 跳过交互，不是代码自动
  帮你同意的。最终真实克隆了一段参考音频、生成了一个 5.2 秒的 wav 文件，用
  `ffprobe` 核实是合法的 24kHz 单声道 PCM 音频。
- `02_STORY_ENGINE/agents/base.py::BaseAgent.generate()`：读完 `02_STORY_ENGINE/prompts/`
  下六份 system prompt 后发现一个结构性漏洞——`Shot.camera`（`CameraSpec`）、
  `shot_size`、`emotion` 这些字段在 `03_STRUCTURED_DATA/schemas.py` 里是闭合枚举
  （`ShotSize` 11 个取值、`CameraAngle` 6 个、`CameraMovement` 13 个、`Emotion` 14 个），
  但六份 prompt 里没有一份完整列出合法取值，只在正文举了几个例子；模型完全靠"猜"确切
  枚举字符串，猜错就触发 Pydantic 校验失败，只能靠最多 3 次重试兜底。对 Claude/GPT-4
  这类强模型问题不大，但对刚接入的免费本地小模型是实打实的可靠性风险。修复：新增
  `_collect_enums()`/`_enum_cheatsheet()`，从 `schema.model_json_schema()` 里递归找出
  所有带 `enum` 的字段定义（覆盖 `$defs` 里的嵌套 schema，比如 `Shot.camera.shot_size`），
  自动拼成"以下字段只能从给定枚举值里精确选一个"的清单，追加进 `generate()` 每次调用
  的 prompt 里——不用去改六个 txt 文件，因为是在 schema 层面动态生成的，以后 schema
  加新枚举字段会自动覆盖到。改完后重新跑了一遍 `02_STORY_ENGINE/demo.py`，确认没有
  破坏 `MockLLMProvider` 靠正则匹配 `[TARGET_SCHEMA=...]` 标记这套机制，六个 scene/shot/
  dialogue/prompt 照常生成成功。
- `13_INFRA/queue/llm_task.py`：原来是直接抛 `NotImplementedError` 的占位 stub，
  现在改成真正调用 `02_STORY_ENGINE.agents.base` 里的 provider——`payload["model"]`
  是 `claude*`/`gpt*` 时走对应的付费 API provider，其他情况（包括不传 `model`）
  默认走免费的 `LocalTransformersProvider`。已在 Celery eager 模式下用真实的小模型
  端到端验证过（问"德国首都是哪"，正确回答"Berlin."）。
- `06_MODELS/{llm,tts,vlm,lipsync}/registry.json` 补充了更多免费/开源候选条目
  （Phi-3.5-mini-instruct、Gemma-2-9b-it、Bark、Bark-small、BLIP-2、Wav2Lip），
  许可证信息不确定的地方明确写"需要自行核实"，没有瞎猜。

## 目前已知、这次没有解决的限制

- **需要真实模型/密钥才能验证的部分只做到"代码对照官方 API 核实 + 离线单测"**：
  Diffusers 的 IP-Adapter/LoRA 角色一致性（`load_ip_adapter`/`set_ip_adapter_scale` 签名
  已对照本机 diffusers 0.40 核实）、Runway 任务轮询、YouTube/TikTok 状态查询与
  Analytics 拉取、Claude/GPT 的 token 用量字段——本机没有 GPU/API key，测试里用假 service
  和 dummy 生成器覆盖了逻辑分支，没有打过真实请求。
- **本机 Homebrew ffmpeg 没有 libass**：字幕无法硬烧，渲染会自动降级为"成片 + SRT 旁路文件"
  并打日志；Docker 镜像里 apt 装的 ffmpeg 带 libass，硬烧路径只在容器里生效。
- `summarize` 压缩策略和 Redis 会话锁只用假 summarizer / 本机临时 redis-server 测过，
  没有用真实 LLM 跑过长会话压缩。
- Python 3.14 + `datasets`/`dill` 不兼容（见上文）——需要等上游修复，或者单独用
  旧版本 Python 的虚拟环境；`pyproject.toml` 已把 `requires-python` 限制为 `<3.14`。
- `11_PUBLISH/reelshort`、`dramabox`、`goodshort` 在这几家平台发布公开开发者 API
  之前，没法做成真实对接——不在本次范围内，抓取真实数据/凭证这次也明确排除在外。
- `07_GENERATION` 里需要付费 API 的 provider（Runway/Pika/ElevenLabs/Sync.so）只
  验证到构造时的凭证检查这一步——这台机器没有对应的 GPU/API key，没法再往下测。
- 对口型（lip-sync）没有实现任何本地免费 provider——目前知名的开源方案要么许可证
  不清楚，要么明确非商用，见上面"免费开源模型选项"表格。
- Coqui XTTS 在项目主环境（`transformers` 5.5.0）里跑不起来（见上文"修复记录"），
  但已经在独立的 `.venv_xtts` 虚拟环境里验证跑通，不算未解决——环境搭建步骤和
  踩过的坑都记在上面"修复记录"和 `tts_generator.py::CoquiXTTSLocalTTSProvider`
  的 docstring 里。`.venv_xtts/` 本身没有加进版本控制（虚拟环境不该提交），需要
  在新机器上按 docstring 里的步骤重新搭一次。
- Mochi 本地视频 provider 只做了接口层面的核实（真实的 diffusers API 签名），没有
  端到端实跑——权重有几十 GB，这台机器的 GPU/显存不够跑。
