"""sfx 子包：一次性音效叠加。导出 SfxEvent（音效文件 + 出现时间点）和 mix_sfx（按时间点叠到成片音轨上）。"""

from .sfx_mix import SfxEvent, mix_sfx

__all__ = ["SfxEvent", "mix_sfx"]
