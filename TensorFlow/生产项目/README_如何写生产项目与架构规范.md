# 如何写生产项目 + 生产架构规范（本目录总纲）

> 这个目录放**综合多技术栈的生产级项目**。先读这份总纲,再看每个项目自己的 `构建顺序.md`。

## 一、写一个生产项目的正确顺序（先想清楚,再动手）
1. **明确需求与指标**：要解决什么问题?怎么算成功?(准确率/延迟 p99/成本/自动化率)。没指标就没法评估。
2. **技术选型（先便宜后贵）**：能用现成模型+提示/RAG 就别微调;要新知识用 RAG,要新能力/格式才微调。选型比调 API 重要。
3. **设计分层架构**：画"依赖方向图"——上层依赖下层,下层不知道上层。定好 core/schemas/services/routers 各放什么。
4. **bottom-up 写代码**：先写没有内部依赖的地基(core),再写上层(services→routers→main)。见各项目 `构建顺序.md`。
5. **写测试（外部依赖要能 stub）**：LLM/模型/DB 用 stub 后端,让测试**离线、确定、秒跑**,不下模型不连网。
6. **容器化 + 运维**：Dockerfile + docker-compose;健康检查、结构化日志、监控、限流、降级兜底。
7. **迭代闭环**：离线评估 → 灰度 A/B → 上线 → 收集 bad case 回流。

## 二、标准生产目录架构（模板,照着套）
```
项目X_名字/
  README.md            # 是什么 / 功能 / 怎么跑 / 架构 / 生产细节
  构建顺序.md           # 按依赖 bottom-up 的写作顺序(照着一个个写)
  requirements.txt     # pip 依赖
  Dockerfile           # 容器镜像
  docker-compose.yml   # 多服务编排(app + 向量库/Redis 等)
  .env.example         # 配置样例(密钥别进仓库)
  run.py               # 本机启动入口
  app/
    core/              # 地基(无内部依赖)
      config.py        #   pydantic-settings 配置(后端可切换/多环境)
      logging.py       #   结构化 JSON 日志
      exceptions.py    #   自定义异常 + 全局处理(统一错误,不泄露栈)
    schemas/           # 接口契约(pydantic 请求/响应,既校验又是文档)
    services/          # 业务核心(一个能力一个文件,可单测)
    routers/           # HTTP 接口层(薄;调 services)
    main.py            # 应用工厂(中间件+异常+路由+静态+启动自检)
    static/            # 前端页面(如有)
  tests/               # 单测(stub 外部依赖,离线可跑)
```
**各层职责一句话**：core=地基 / schemas=契约 / services=业务(重) / routers=薄接口 / main=组装。
**铁律**：依赖只能从上往下(routers→services→core),反过来不行;routers 里不写业务,业务全在 services。

## 三、为什么 bottom-up 写（依赖方向）
上层 `import` 下层,所以**下层先写上层才有东西可 import**。顺序：
`config → logging → exceptions → schemas → services(按内部依赖排) → routers → main → tests`。

## 四、生产 checklist（每个项目都要过）
- **安全**：输入长度/类型校验;subprocess 用列表参数不用 shell;密钥走环境变量;防注入。
- **性能**：模型/连接**单例 + 懒加载**;相同输入**内容哈希缓存**;批处理。
- **并发**：慢操作(训练/视频/大文档)走**异步 job + 轮询**,线程池限并发。
- **健壮**：启动自检依赖;缺依赖降级可用而非崩;统一异常;超时/重试/熔断/降级兜底。
- **清理**：临时文件用完即删(成功失败都删);缓存/任务 TTL 过期清理。
- **可切换**：外部依赖(LLM/嵌入/向量库)抽象成"后端",stub/本地/云可配置切换 → 测试能离线。
- **可观测**：结构化日志 + request_id;监控 p99/QPS/错误率/资源;健康检查。
- **配置**：全走环境变量;多环境;`.env.example` 给样例。

## 五、本目录项目清单（覆盖各课程域,统一架构,stub 后端离线可验证）
| 项目 | 课程域 | 综合技术栈 |
|------|--------|-----------|
| 项目1_企业知识库问答平台 | transformers/RAG/Agent | 解析+切块+嵌入+检索+重排+LLM+Agent+缓存+异步+前端+Docker |
| 项目2_内容审核中台 | NLP 应用 | 敏感词/PII检测+风险打分+分级+脱敏+异步批量 |
| 项目3_LLM推理网关 | 推理部署 | 多后端路由+降级兜底+缓存+限流+OpenAI接口 |
| 项目4_微调与对齐平台 | 训练/对齐 | 数据校验+SFT(可切trl)+评估+模型注册+异步训练 |
| 项目5_视觉推理服务 | Computer Vision | 分类/检测(可切transformers) |
| 项目6_语音服务 | Audio | ASR 转写 + TTS 合成 |
| 项目7_Agent编排服务 | Agent/Context | 手写 ReAct 循环 + 工具 |
| 项目8_MCP工具服务器 | MCP | 工具契约 + tools/list + tools/call |
| 项目9_图像生成服务 | Diffusion | 文生图(可切 StableDiffusion) |
| 项目10_3D生成服务 | ML for 3D | 文本/图→3D 高斯(可切 LGM) |
| 项目11_机器人策略服务 | Robotics | 观测→动作(可切 LeRobot 策略) |
| 项目12_RL训练服务 | Deep RL | 训练奖励曲线+评估(可切 gym+SB3) |
| 项目13_实时流式对话服务 | 推理/部署 | WebSocket 逐token流式(打字机) |
| 项目14_多模态图文检索 | CV+NLP 多模态 | CLIP式图文同空间检索 |
| 项目15_LLM评测服务 | 评估/MLOps | 准确率/幻觉率/LLM-as-judge |

> 统一设计:外部依赖(模型/GPU)抽象成"后端",默认 **stub → 零下载离线端到端可测**;生产切 real/hf/openai/trl/mlx。
> 每个项目都有 `构建顺序.md`(bottom-up)。更多项目按同架构继续加。
