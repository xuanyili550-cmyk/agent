"""music 子包：背景音乐混音。只导出 mix_bgm，把 BGM 压低音量后垫在对白音轨下面。"""

from .music_mix import mix_bgm

__all__ = ["mix_bgm"]
