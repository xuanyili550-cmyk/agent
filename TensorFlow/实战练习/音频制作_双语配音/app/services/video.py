"""
================================================================================
 服务层 · 视频合成（每句一帧 · 英文词上标 IPA · 底部进度条 · 逐句拼成 MP4）
================================================================================
 【这个文件做什么】把文本做成一个"学习视频"：每句话渲染成一张画面(英文词上方标音标、
   中文内联)，配上这句的音频，底部一条随播放时间增长的进度条(视频感/丝滑)，逐句拼成 mp4。

 【为什么是"每句一张静帧"而不是逐帧动画】
   `say` 给不出词级时间戳，做不到"逐词卡拉OK对齐"。工程上取"句级"粒度：一句 = 一张静帧 + 该句音频，
   再用 ffmpeg 的 drawbox 画一条【随时间 t 增长】的进度条(w=iw*t/时长)，就有了流畅的推进感，
   且鲁棒、开销小。首尾各 0.2s 淡入淡出，让句与句之间过渡不生硬。

 【关键技术点】
   · PIL 排版：按画面宽度把 token 折行；英文词上方用小字号画 IPA，词本身居中(ruby 效果)。
   · 字体：必须用含【中文 + IPA 音标】字形的字体(如 Arial Unicode)，否则音标/中文会变成豆腐块。
   · ffmpeg 一段视频 = -loop 1 静图 + 音频 + drawbox 进度条滤镜；各段再 concat 成整片。
   · 缓存 & 清理：整片按内容哈希缓存；所有中间帧/分段用完即删(失败也删)。
================================================================================
"""
import os
import shutil
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont

from ..core.config import get_settings
from ..core.exceptions import AppError, ValidationError
from . import cache, tts
from .system import resolve_font
from .tokenizer import analyze


def _hex(c: str) -> str:
    """'#38bdf8' → '0x38bdf8'：ffmpeg 滤镜里的颜色写法。"""
    return "0x" + c.lstrip("#")


def _rgb(c: str):
    """'#0f172a' → (15,23,42)：PIL 要的 RGB 元组。"""
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def _fonts(font_path: str):
    """三档字号：标题/正文/音标。音标用小字号画在词的上方。"""
    return {
        "title": ImageFont.truetype(font_path, 34),
        "main": ImageFont.truetype(font_path, 52),
        "ipa": ImageFont.truetype(font_path, 26),
    }


def _wrap(tokens, font_main, font_ipa, max_w):
    """把一句的 token 排成多行(每行是若干 token)。
    英文词的占位宽度要取【词宽 和 音标宽 的较大值】，否则音标会跟相邻词挤到一起。"""
    lines, cur, cur_w = [], [], 0
    dummy = Image.new("RGB", (10, 10))
    d = ImageDraw.Draw(dummy)                        # 借一个 Draw 来量文字宽度
    for t in tokens:
        w_main = d.textlength(t["text"], font=font_main)
        w_ipa = d.textlength(t.get("ipa", ""), font=font_ipa) if t["kind"] == "en" else 0
        w = max(w_main, w_ipa) + 14                  # +14 给词间留点空隙
        if cur and cur_w + w > max_w:                # 超出行宽 → 换行
            lines.append(cur)
            cur, cur_w = [], 0
        cur.append((t, w))
        cur_w += w
    if cur:
        lines.append(cur)
    return lines


def _render_frame(sentence, idx, total, font_path, out_png):
    """渲染一句的画面：右上角页码，正文居中；英文词上方标 IPA(音标用强调色)。"""
    s = get_settings()
    W, H = s.video_width, s.video_height
    img = Image.new("RGB", (W, H), _rgb(s.video_bg))
    d = ImageDraw.Draw(img)
    F = _fonts(font_path)
    fg, accent = _rgb(s.video_fg), _rgb(s.video_accent)
    muted = tuple(int(x * 0.6 + 30) for x in fg)     # 页码用暗一点的颜色

    d.text((60, 40), f"{idx + 1} / {total}", font=F["title"], fill=muted)

    lines = _wrap(sentence["tokens"], F["main"], F["ipa"], W - 160)   # 左右各留 80 边距
    line_h = 52 + 26 + 24                             # 每行高 = 正文 + 音标 + 行距
    total_h = len(lines) * line_h
    y = max(140, (H - total_h) // 2)                 # 整块垂直居中(顶部至少留 140 给页码)
    for line in lines:
        line_w = sum(w for _, w in line)
        x = (W - line_w) // 2                         # 每行水平居中
        for t, w in line:
            if t["kind"] == "en" and t.get("ipa"):
                iw = d.textlength(t["ipa"], font=F["ipa"])
                d.text((x + (w - iw) / 2, y), t["ipa"], font=F["ipa"], fill=accent)   # 音标在上、居中
            mw = d.textlength(t["text"], font=F["main"])
            color = accent if t["kind"] == "en" else fg     # 英文词用强调色，突出学习对象
            d.text((x + (w - mw) / 2, y + 34), t["text"], font=F["main"], fill=color)  # 词在下
            x += w
        y += line_h
    img.save(out_png)


def _segment(png, wav, dur, out_mp4):
    """静帧 + 音频 → 一段 mp4。
    drawbox 画进度条：x=0 顶到左边，w=iw*t/时长 随播放时间 t 线性增长到满宽 → 丝滑推进。
    fade 首尾各 0.2s 淡入淡出；format=yuv420p 保证各种播放器都能播。"""
    s = get_settings()
    bar = (f"drawbox=x=0:y=ih-12:w=iw*t/{dur:.3f}:h=12:color={_hex(s.video_accent)}:t=fill,"
           f"fade=t=in:st=0:d=0.2,fade=t=out:st={max(0, dur - 0.2):.3f}:d=0.2,format=yuv420p")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", png, "-i", wav,
                    "-vf", bar, "-r", str(s.video_fps), "-c:v", "libx264", "-tune", "stillimage",
                    "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p", "-t", f"{dur:.3f}",
                    "-shortest", out_mp4], check=True, capture_output=True)


def build(text: str, out_mp4: str, progress_cb=None) -> str:
    """整段文本 → mp4。命中缓存直接返回；progress_cb(done,total) 汇报进度(给前端进度条)。"""
    s = get_settings()
    font_path = resolve_font()                       # 拿含中文+IPA 字形的字体(缺则报依赖错)
    key = cache.key_for("mp4", text, {"w": s.video_width, "h": s.video_height, "fps": s.video_fps})
    hit = cache.get(key, "mp4")
    if hit:
        if progress_cb:
            progress_cb(1, 1)
        return hit                                   # 命中缓存 → 秒回

    data = analyze(text)                             # 复用分词服务：切句 + 英文标 IPA
    sents = data["sentences"]
    if len(sents) > s.max_export_sentences:          # 句数上限：保护时长/机器资源
        raise ValidationError(f"导出句数过多(>{s.max_export_sentences})，请分批。")

    work = tempfile.mkdtemp(prefix="video_")
    try:
        segs = []
        total = len(sents)
        for i, sent in enumerate(sents):             # 逐句：渲帧 → 合音 → 出段
            png = os.path.join(work, f"f{i}.png")
            wav = tts.synth_wav(sent["text"])        # 这一句的音频(可能命中缓存)
            dur = max(0.6, tts.duration(wav))        # 太短的句子给个下限，避免闪一下
            _render_frame(sent, i, total, font_path, png)
            seg = os.path.join(work, f"s{i}.mp4")
            _segment(png, wav, dur, seg)
            segs.append(seg)
            if progress_cb:
                progress_cb(i + 1, total)            # 每出一段就汇报进度
        if not segs:
            raise AppError("无可导出内容。")
        # 所有段 concat 成整片
        listf = os.path.join(work, "list.txt")
        with open(listf, "w") as f:
            for p in segs:
                f.write(f"file '{p}'\n")
        out = cache.path_for(key, "mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                        "-i", listf, "-c", "copy", out], check=True, capture_output=True)
        return out
    except subprocess.CalledProcessError as e:
        raise AppError(f"视频合成失败：{(e.stderr or b'').decode('utf-8', 'ignore')[:200]}")
    finally:
        shutil.rmtree(work, ignore_errors=True)      # ★ 清理所有中间帧/分段
