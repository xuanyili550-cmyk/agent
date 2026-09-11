# AI_SHORT_DRAMA 架构文档

这份文档写给"自己动手重写一遍"用的——不讲某个文件第几行写了什么（那是 README.md 的
事），讲的是**为什么整个系统要这样分层、核心设计决策是什么**。手写代码时卡住了、
忘了某个模块该负责什么，回来看这份文档。

## 1. 一句话概括

这是一条把"一个故事创意"自动变成"一集可以发布的竖屏短剧视频"的流水线，中间每一步
都是"结构化数据 -> 下一步结构化数据"的转换，最后一步才落地成视频文件。

```
故事创意（自然语言）
   → Story Bible / Character / Episode（结构化 JSON）
   → Scene / Shot / Dialogue（结构化 JSON，颗粒度细到每一个镜头）
   → 图片 / 视频 / 语音（二进制文件 + 一条 AssetRecord 元数据）
   → QC 打分（结构化 JSON：通过 / 重试）
   → 剪辑渲染（Episode.mp4）
   → 发布 + 数据回收（结构化 JSON：留存/点击率/收入）
```

## 2. 核心设计决策（为什么这样分层）

### 2.1 Shot 是全系统唯一的关联键

一集短剧最终会被拆成几十个几秒钟的镜头（Shot）。从故事生成到最后渲染，中间每一层
都是"围绕 Shot 展开的工作"：

- `03_STRUCTURED_DATA.Shot.id` 生成出来之后
- `07_GENERATION` 生成的每张图/每段视频，AssetRecord 里都要记 `shot_id`
- `08_QC` 的打分报告按 `shot_id` 记
- `10_EPISODES` 的 manifest 里 `shot_ids` 汇总整集用到的所有镜头

**为什么这么设计**：短剧是"镜头级"生产，不是"整集打包生成"。只有把 Shot 作为
最小追踪单元，才能做到"这一个镜头质量不行 -> 只重新生成这一个镜头 -> 不用整集重来"。
如果你自己写的时候不small心把 Shot 的 id 设计漏了、或者中途换了别的字段当 key，
后面 QC/重试/资产管理这些环节全部会散架——**这是整个架构里最不能省的一步**。

### 2.2 03_STRUCTURED_DATA 必须是唯一真源（Single Source of Truth）

`Shot`、`Character`、`Episode` 这些模型只能在一个地方定义一次（`03_STRUCTURED_DATA/
schemas.py`），其他所有模块（数据集、生成、QC、发布）都从这里 import，不能自己重新
定义一份。

**为什么**：这份参考实现踩过这个坑——4 个并行开发的模块里，有一个因为开工时
schemas.py 还没写出来，自己臆造了一份字段名不一样的 Shot 模型，后来两边对不上，
排查花了不少功夫（细节见 `README.md` 的"修复记录"）。你自己一个人写不存在"并行
开发互相看不到代码"这个问题，但同样的错误换一种方式也会发生：写着写着在另一个文件
里想"省事就地定义一个简化版"，结果就是同一个概念在系统里有两份不一致的定义。
**规则：需要 Shot/Character/Episode 的字段，永远从 03_STRUCTURED_DATA import，
不要重新写。**

### 2.3 每一个"生成"模块都用 Provider 抽象，不要直接在业务逻辑里调 API

`07_GENERATION` 下 image/video/audio/voice/lipsync 每一类都是：
一个抽象基类（定义 `generate()`/`synthesize()`/`sync()` 接口）+
若干具体 Provider 实现（本地免费模型 or 付费 API）。上层代码永远只依赖抽象基类，
不关心当前用的是本地 Diffusers 还是 Runway 的付费 API。

**为什么**：这样才能做到"没有 API key 时用免费本地模型跑通全流程做开发/测试，
有预算之后换成付费 API 提升质量"，换的时候只改一行"传哪个 Provider 进去"，业务逻辑
（怎么读 Shot、怎么写 AssetRecord）完全不用动。如果你直接在业务代码里
`requests.post("https://api.xxx.com/...")`，以后想换供应商或者想先用免费方案过渡，
就要把调用逻辑到处改。

### 2.4 结构化输出用 Pydantic 校验 + 重试，不要相信 LLM 输出的格式

`02_STORY_ENGINE/agents/base.py::BaseAgent.generate()` 的模式是：LLM 输出文本 ->
`schema.model_validate_json()` 校验 -> 校验失败就把报错信息喂回去重试，最多 N 次。

**为什么**：LLM 不是100%可靠的 JSON 生成器，尤其是字段是"闭合枚举"（比如镜头景别
只能是 `close_up`/`medium_shot` 这几个固定值之一）的时候，模型很容易输出一个语义
对但字符串不对的值。手写这部分时容易漏掉两件事：① 重试要把上一次的报错信息带给
模型，不能只是"再问一遍一模一样的问题"；② 如果 schema 里有闭合枚举字段，最好把
合法取值清单直接写进 prompt 里（这套参考实现是从 `schema.model_json_schema()` 里
动态抽取的，见 `02_STORY_ENGINE/agents/base.py::_enum_cheatsheet()`），不要指望模型
自己"猜对"。

### 2.5 数据库只存元数据，大文件进对象存储

`13_INFRA` 的 Postgres 表（Asset 表）只存 `file_path`/`character_id`/`model`/
`prompt`/`seed`/`license`/`created_at` 这些元数据，图片/视频/音频本身不进数据库，
进本地磁盘或 S3/R2（`13_INFRA/storage`）。

**为什么**：短剧生产会产生海量二进制文件（一集几十个镜头 × 图片 + 视频 + 音频），
塞数据库会让数据库迅速膨胀且备份/查询都变慢。这是后端工程里的常识，但从头写的时候
容易图省事直接把文件塞进数据库字段，一开始感觉不出问题，数据量上来就是灾难。

### 2.6 队列任务只做"调度 + 落盘 + 记账"，不做业务判断

`13_INFRA/queue/*.py` 里的每个 task（image_task/video_task/...）职责很窄：接收
payload -> 调用对应 Provider -> 存文件 -> 写 AssetRecord -> 返回结果。不在 task 里
判断"这个镜头要不要重试""质量够不够"——那是 `08_QC` 的职责。

**为什么**：GPU worker 是稀缺资源，task 的职责必须单一、可预测，这样才能水平扩展
（1 张 GPU 到 100 张 GPU 架构不用改，见原始需求里的说法）。如果把 QC 判断逻辑塞进
生成 task 里，task 就变得又慢又难测试，扩展的时候还得考虑"这个 worker 到底在干嘛"。

## 3. 分层职责一览

| 层 | 输入 | 输出 | 不该做的事 |
|---|---|---|---|
| 01_CONTENT | 人工创意 | Markdown | 不该有代码逻辑，纯内容 |
| 02_STORY_ENGINE | 故事创意 | 03 的结构化 JSON | 不该直接调用生成模型（图/视频/语音） |
| 03_STRUCTURED_DATA | - | Pydantic schema + 示例数据 | 不该包含任何业务逻辑，只是数据契约 |
| 04_DATASET | 参考素材 | HF datasets manifest | 不该抓取真实版权数据 |
| 05_TRAINING | 04 的数据集 | 训练好的 LoRA/checkpoint | 不该在没有说清楚数据来源许可证的情况下训练 |
| 06_MODELS | - | 模型注册表（谁在用哪个 checkpoint、许可证） | 不该是训练/推理代码本身，只是登记表 |
| 07_GENERATION | Shot | 图片/视频/音频文件 + AssetRecord | 不该做 QC 判断、不该做后期剪辑 |
| 08_QC | 生成的素材 | QC 报告（通过/重试） | 不该负责重新生成，只负责打分和给判定 |
| 09_POST | 通过 QC 的素材 | Episode.mp4 | 不该负责生成素材本身 |
| 10_EPISODES | 09 的产物 | episode_manifest.json | 不该包含渲染逻辑，只是清单 |
| 11_PUBLISH | 10 的 manifest | 发布结果 | 不该包含内容生成/剪辑逻辑 |
| 12_ANALYTICS | 平台回传的数据 | 留存/CTR/收入报表 | 不该反向修改内容生产流程（那是人看报表后自己决定） |
| 13_INFRA | 横切 | API/DB/队列/存储 | 不该包含任何具体的业务判断逻辑（生成质量好坏、发布策略等） |

## 4. 技术选型速查（详细版本号见各目录 requirements）

| 需求 | 选型 | 为什么 |
|---|---|---|
| 结构化数据契约 | Pydantic v2 | 校验 + 类型提示 + `model_json_schema()` 能动态导出给 LLM |
| Agent 编排 | LangGraph | 把多个 Agent 串成有状态的图，比手写一堆 if/else 调用链清楚 |
| 免费 LLM | 本地 transformers pipeline | 不需要 API key 就能跑通开发/测试 |
| 免费图片生成 | Diffusers（SDXL/FLUX） | 本地跑，无需付费 API |
| 后端 | FastAPI + PostgreSQL + Redis + Celery | 标准的"API + 关系数据库 + 任务队列"组合，GPU worker 天然适合用队列削峰 |
| 后期合成 | FFmpeg（subprocess 调用，不引入额外剪辑框架） | 命令行工具足够稳定，不需要为剪辑单独造轮子 |

## 5. 手写时最容易踩的坑（这套参考实现真实踩过的）

1. **枚举字段的合法值一定要么写进 prompt，要么做成 schema 层面动态生成**——不要
   假设 LLM 能"猜对"闭合枚举的确切字符串。
2. **免费本地模型的 API 用法容易凭印象写错**——比如 Bark 的 voice 预设参数应该走
   `preprocess_params={"voice_preset": ...}` 而不是 `forward_params=
   {"history_prompt": ...}`，这种细节最好实际跑一次确认，不要只读文档就下结论。
3. **第三方库版本冲突是常态**——同一个 Python 环境里塞太多深度学习库，版本冲突
   概率很高（这套实现就遇到过 `datasets`/`dill` 在 Python 3.14 下崩溃、
   `coqui-tts` 和 `transformers` 5.x 不兼容）。遇到这种问题**该开独立虚拟环境就
   开独立虚拟环境**，不要为了"一个环境搞定一切"死磕版本号。
4. **数字开头的目录名不能被 `import` 语句直接引用**——用 `importlib.import_module`
   带完整点号路径，不要图省事把子目录塞进 `sys.path` 再裸 import（这样多个同名
   文件会互相覆盖，比如这个项目里就有两个 `schemas.py`）。
5. **许可证信息不要凭记忆断言**——写模型注册表时，能查的（Hugging Face Hub API
   的 `license` 标签）就去查，查不到的老实写"需要自行核实"，不要为了看起来完整
   就编一个听起来合理的许可证名字。

## 相关文档

- `BUILD_ORDER.md` —— 建议的分阶段实现顺序 + 每阶段验收清单
- `README.md` —— 这套参考实现本身的现状记录（哪些真实可跑、哪些是占位、踩过什么坑）
