# 建议的动手实现顺序

配合 `ARCHITECTURE.md` 一起看。这份文档回答"第一步该建什么、建完要能跑通什么、
怎么知道自己写对了"，按阶段推进，不要一上来就想把 13 个模块同时铺开。

**先决定一件事：你的代码写在哪。** 建议不要直接在这个 `AI_SHORT_DRAMA/` 目录里改，
新建一个平级目录（比如 `AI_SHORT_DRAMA_practice/`）自己从空目录开始写，卡住了再回来
对照这边的参考实现看某个文件是怎么处理的。这样两边可以随时 diff 对比，也不会把
现成的参考实现改坏。

每个阶段结束时都有一条"验收标准"——**必须真的跑起来看到预期结果才算这阶段完成**，
不要读完代码觉得"逻辑上应该没问题"就往下走，这套参考实现前面几轮就是靠"每步都实跑
验证"才挖出了好几个真实 bug（枚举值没告诉 LLM、Bark 的参数名传错、库版本冲突……），
不实跑是发现不了这些的。

## Phase 0：骨架

- [ ] 建 13 个顶层目录（参考 `AI_SHORT_DRAMA/` 现有的目录树，`ls AI_SHORT_DRAMA` 抄
      一遍结构，不用抄内容）
- [ ] 建 `requirements.txt`（先留空，每个 Phase 写到哪个包就加哪个，不要一开始把
      所有依赖都装上——用不到的依赖装了你也不知道是干嘛的）

## Phase 1：03_STRUCTURED_DATA —— 先定数据契约，这是地基

这是全项目唯一必须"先做"的一步，其他所有模块都依赖这里的模型定义。

- [ ] `schemas.py`：用 Pydantic v2 定义 `StoryBible`、`Character`、`Episode`、
      `Scene`、`Shot`、`DialogueLine`、`ImagePrompt`、`VideoPrompt`
- [ ] 想清楚 `Shot` 的字段（至少要有：唯一 id、属于哪个 episode/scene、出场角色、
      地点、动作描述、情绪、镜头参数、时长）——这是全系统的关联键，字段设计一旦定下来
      后面所有模块都会依赖它，改动代价很高，值得多花点时间想清楚再往下写
- [ ] 给会重复出现固定取值的字段（镜头景别、情绪、机位角度……）定义成 `Enum`，
      不要用自由字符串——后面 Phase 2 会用到这些枚举
- [ ] 手写一份示例数据（一集完整的 Story Bible + 几个 Scene/Shot），用
      `Model.model_validate_json()` 验证能通过

**验收**：`python -c "import schemas; schemas.Shot.model_json_schema()"` 不报错；
手写的示例 JSON 能被对应模型 `model_validate_json()` 成功解析。

## Phase 2：02_STORY_ENGINE —— 先用假 LLM 跑通结构化生成链路

**不要一上来就接真实 LLM API**——先证明"prompt -> LLM 输出 -> Pydantic 校验 ->
拿到强类型对象"这条链路本身是通的，再考虑换真实模型。

- [ ] 写一个 `LLMProvider` 抽象接口（一个 `complete(system_prompt, user_prompt) ->
      str` 方法就够）
- [ ] 写一个 `MockLLMProvider`：不调用任何真实模型，直接返回你手写的固定 JSON 字符串
- [ ] 写 `generate()` 通用逻辑：调用 provider -> 校验 -> 校验失败就把报错信息拼回
      prompt 里重试，最多 N 次
- [ ] 只写一个 Agent（比如 `StoryAgent.generate_story_bible(idea)`），跑通
      "故事创意 -> StoryBible 对象"这一步
- [ ] 验收通过后，再依次加其他 5 个 Agent（Character/Episode/Screenplay/
      Storyboard/Prompt），每加一个都单独跑通再加下一个，不要一次性全写完再一起调试

**验收**：写一个 `demo.py`，用 `MockLLMProvider` 从一句话故事创意，走到生成出
`Shot`/`ImagePrompt` 列表，全程不需要任何 API key，能直接 `python demo.py` 跑通。

**（可选，等上面跑通了再做）** 接入真实 LLM：写 `AnthropicProvider`/
`OpenAIProvider`，或者本地免费模型的 Provider（`transformers.pipeline
("text-generation", ...)`），确认没有 API key 时会抛出清晰的报错而不是静默失败。

## Phase 3：04_DATASET —— manifest + loader

- [ ] 写一个生成占位图片的小工具（PIL 画纯色 + 文字），不要一上来就想着抓真实素材
- [ ] 定义 manifest 的 JSONL 格式（比如 character_manifest.jsonl 每行
      `{character_id, image_path, caption}`）
- [ ] 写 loader：用 Hugging Face `datasets` 库把 manifest 加载成 `Dataset` 对象

**验收**：`build_character_dataset()` 返回的 `Dataset` 行数、字段跟你写的 manifest
对得上。

## Phase 4：07_GENERATION —— 先占位生成器，再接真实模型

- [ ] 定义 `BaseImageGenerator` 抽象类
- [ ] 写 `DummyImageGenerator`：不调用任何模型，画一张带 prompt 文字的占位图
- [ ] 写 `AssetRecord`（字段：asset_id、file_path、character_id、episode_id、
      shot_id、model、prompt、seed、license、created_at）+ 写入 jsonl 的函数
- [ ] 写 `demo.py`：读 Phase 1 里的 Shot/ImagePrompt 数据 -> 生成占位图 -> 写
      AssetRecord，确认 `shot_id` 能对得上 Phase 1 的 Shot.id

**验收**：跑完 demo 后检查 asset_records.jsonl 里的 `shot_id` 字段，和 Phase 1
数据里的 Shot id 完全一致——这一步是在验证"Shot 关联键"这个核心设计有没有断掉。

**（可选）** 加真实 Provider（本地 Diffusers 或付费 API），每加一个都要真的跑一次
（哪怕只生成一张图），不要只读官方文档代码就假设写对了——这套参考实现光是接 Bark
TTS 就因为参数名传错踩了一个坑，实跑才发现。

## Phase 5：08_QC —— 用 CLIP 做一致性打分

- [ ] 用 `transformers` 的 CLIP 模型，写一个"计算两张图片相似度"的函数
- [ ] 拿一张图片自己和自己比一次（应该接近 1.0），再拿两张完全不相关的图比一次
      （应该明显更低），确认打分逻辑本身是对的
- [ ] 定义 QC 报告结构（各项分数 + 阈值 + 通过/重试判定）

**验收**：同图自比分数明显高于随机图对比分数——这是最基础的"打分逻辑没写反"验证，
非常便宜但很容易漏掉。

## Phase 6：09_POST —— ffmpeg 拼接渲染

- [ ] 用 `subprocess` 调 ffmpeg，把几张占位图片（或几秒钟的纯色测试视频片段）拼接成
      一个视频文件
- [ ] 加字幕烧录、裁剪成 9:16 竖屏

**验收**：用 `ffprobe` 检查输出文件的时长、分辨率、宽高比是否和预期一致——不要只看
"命令跑完没报错"，要检查产物的实际参数。

## Phase 7：13_INFRA —— 用后端把前面几层串起来

到这一步前面几层都应该已经能各自独立跑通了，这一层负责把它们接进一个真实的系统。

- [ ] SQLAlchemy 模型（Asset 表字段对齐 Phase 4 的 AssetRecord）
- [ ] FastAPI 的基础 CRUD 路由
- [ ] Celery task：接收一个 Shot 的 payload -> 调用 Phase 4 的 Generator -> 存
      AssetRecord，跑在 eager 模式下（不需要真实 Redis）

**验收**：写一个 CRUD 往返测试（建一条记录 -> 查出来 -> 删掉 -> 确认查不到了），
不需要真实 Postgres，用 SQLite 就够。

## Phase 8：10_EPISODES / 11_PUBLISH / 12_ANALYTICS —— 收尾

- [ ] Episode manifest：汇总一集用到的所有 shot_id、渲染产物路径、QC 结果
- [ ] 发布客户端：先写抽象接口 + `--dry-run` 模式（只校验 payload 结构，不真的发请求）
- [ ] 数据分析：读一份手写的合成事件数据（jsonl），算留存率/点击率

**验收**：`--dry-run` 模式下发布客户端不报错；分析脚本在合成数据上算出的数字符合
预期（比如你手写 10 个用户、5 个第二天回来，D1 留存算出来应该是 50%）。

## Phase 9（最后做，甚至可以不做）：05_TRAINING / 06_MODELS

**先不要训练自己的模型**——这是原始需求文档里就写明的建议："第一版直接：API LLM ->
结构化 JSON，跑通以后，再用你自己的剧本数据做 SFT / LoRA"。等 Phase 1-8 全部跑通、
你自己攒了一些真实的剧本/角色数据之后，再回来做这一步：

- [ ] 模型注册表：记录你在用哪个模型、什么许可证（能查的去 Hugging Face Hub API 查，
      不要凭记忆瞎写）
- [ ] 训练脚本模板（LoRA/SFT），先跑通脚本本身的参数解析和数据加载逻辑，不追求真的
      跑一次完整训练

## 每阶段之间的自查清单

每完成一个 Phase，问自己三个问题：

1. **能不能不看代码，直接跑一条命令看到正确结果？**（Phase 里给的"验收"标准）
2. **这一层有没有偷偷依赖了还没定义清楚的东西？**（比如 Phase 2 写 Shot 相关逻辑时，
   字段名有没有跟 Phase 1 的 schemas.py 完全对齐）
3. **如果换一个人接手这段代码，他能不能只看这一个目录就知道该传什么、拿到什么？**
   （每层的输入输出边界要清楚，别的层不该关心这层内部怎么实现的）

## 相关文档

- `ARCHITECTURE.md` —— 为什么要这样分层、核心设计决策
- `README.md` —— 参考实现本身的现状记录（可以在某个 Phase 卡住时过去对照）
