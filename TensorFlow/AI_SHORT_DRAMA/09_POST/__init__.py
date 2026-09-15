"""09_POST 后期制作包：把生成好的镜头素材（视频/关键帧、对白、BGM、音效）合成为一集成片 Episode.mp4。

在流水线里位于 07_GENERATION（素材生成）/ 08_QC 之后、10_EPISODES（成片管理）/ 11_PUBLISH（发布）之前。
子包分工：editing（拼接）、rendering（9:16 适配 + 总编排）、subtitle（字幕）、music（BGM）、sfx（音效）；
assemble.py 是面向上游的入口，把"镜头素材列表"翻译成 rendering 需要的配置。
本包本身不导出符号，各子模块通过 importlib.import_module("09_POST.xxx") 按需加载（目录名以数字开头，不能直接 import）。
"""
