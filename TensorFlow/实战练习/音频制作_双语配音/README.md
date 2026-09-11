# 双语配音 · 音频/视频制作服务（生产级）

中英混排文本 → **英文自动标 IPA 音标** → **中英文 TTS 朗读**（逐句卡拉OK高亮）→ **一键导出 MP4**。
前端页面 + FastAPI 后端 API，分层工程结构，本机零模型下载即可跑。

## 快速开始
```bash
cd 音频制作_双语配音
pip install -r requirements.txt        # fastapi/uvicorn/pydantic/pillow
# 系统依赖：brew install ffmpeg ；macOS `say` 系统自带
python run.py                          # → http://127.0.0.1:8000  (Swagger: /docs)
```
打开 http://127.0.0.1:8000：输入文本 → 分析（看音标）→ ▶ 播放全部 / 点词点句发音 → 🎬 导出 MP4。

## 架构（对齐仓库 生产级项目_NL2SQL 的分层）
```
app/
  core/     config(pydantic-settings) · logging(JSON行) · exceptions(全局处理)
  schemas/  audio(请求/响应 pydantic 契约)
  services/ system(声音/字体/ffmpeg 探测) · tokenizer(分句分词+IPA)
            ipa(eng_to_ipa→内置词典→规则兜底) · tts(say分段+ffmpeg拼接+缓存)
            video(PIL帧+ffmpeg进度条→逐句mp4) · cache(内容哈希) · jobs(异步任务池)
  routers/  health · audio(/analyze /tts) · video(/export /jobs /download)
  main.py   应用工厂：CORS+异常+路由+静态前端+启动自检
  static/   index.html · styles.css · app.js（主题/三模式/高亮/导出进度）
tests/      ipa · tokenizer · api
run.py      uvicorn 入口
```

## 生产细节（已处理）
- **安全**：`say -f 文件` 传文本、subprocess 全列表参数（无 shell 注入）；输入长度/句数/词长上限。
- **性能**：内容哈希缓存——相同文本的音频/视频只生成一次；英文 IPA 带 LRU。
- **并发**：视频导出 CPU 密集 → 异步 job + 轮询进度，线程池限并发，保护机器。
- **健壮**：启动自检 say 声音/CJK+IPA 字体/ffmpeg；缺依赖 `/health` 显示 degraded 并明确报错。
- **清理**：临时目录用完即删（失败也删）；缓存与 job 过期清理。
- **可切换**：TTS 默认 macOS `say`，换 mlx_audio/云 TTS 只改 `services/tts._say_segment` 一处。

## 配置（环境变量 TTSAPP_*）
`TTSAPP_ZH_VOICE`(默认 Tingting) `TTSAPP_EN_VOICE`(Samantha) `TTSAPP_SPEECH_RATE`(175)
`TTSAPP_VIDEO_WIDTH/HEIGHT/FPS` `TTSAPP_MAX_TEXT_CHARS` `TTSAPP_MAX_EXPORT_SENTENCES` `TTSAPP_CACHE_DIR`

## 音标精度
内置常用词词典准确；生僻词为规则近似（前端标注“近似”）。装 `pip install eng-to-ipa` 后**自动切换**为全量准确。

## 测试
```bash
pip install pytest && pytest        # ipa/tokenizer/api（音频/视频为集成，需系统 say+ffmpeg）
```

## TTS 后端切换（高音质 vs 零下载）
本服务支持两个 TTS 后端，环境变量 `TTSAPP_TTS_BACKEND` 切换：

| 后端 | 音质 | 依赖 | 适用 |
|------|------|------|------|
| `mlx`(默认) | **高·自然**(Kokoro-82M 神经 TTS) | `pip install mlx-audio`；首次下模型 ~几百 MB | Apple Silicon，追求音色 |
| `say` | 一般(系统机械音) | 零下载(macOS 自带) | 快速/离线/无网 |

- **自动兜底**：默认走 mlx；若 mlx 模型下不动或接口不符，会自动【回退到 say】，服务不中断。
- mlx 音色/语种码可配：`TTSAPP_MLX_EN_VOICE`(af_heart) `TTSAPP_MLX_ZH_VOICE`(zf_xiaobei)
  `TTSAPP_MLX_EN_LANG`(a) `TTSAPP_MLX_ZH_LANG`(z) `TTSAPP_MLX_SPEED`(1.0)。
  若报语种码错误，把 lang 改成 `en`/`zh` 试(不同 mlx-audio 版本约定不同)。
- 想先用零下载版验证：`TTSAPP_TTS_BACKEND=say python run.py`。
