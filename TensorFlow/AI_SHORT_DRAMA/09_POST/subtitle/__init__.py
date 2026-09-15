"""subtitle 子包：字幕生成与硬烧。

导出 SubtitleCue（一条字幕）、两种 cue 来源（流水线对白 JSON / Whisper ASR 结果）、write_srt 和 burn_subtitles。
SubtitleBurnUnavailable / ffmpeg_has_filter 不在这里导出，需要时直接从 .subtitle 取（rendering 就是这么做的）。
"""

from .subtitle import (
    SubtitleCue,
    burn_subtitles,
    cues_from_dialogue_json,
    cues_from_whisper_result,
    write_srt,
)

__all__ = [
    "SubtitleCue",
    "burn_subtitles",
    "cues_from_dialogue_json",
    "cues_from_whisper_result",
    "write_srt",
]
