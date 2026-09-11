"""
================================================================================
 服务层 · 文本分析（分句 + 分词 + 英文标 IPA + 输入校验）
================================================================================
 【这个文件做什么】把一整段中英混排文本，拆成"句子 → token(英文词/中文字/标点/空白)"，
   并给每个英文词查好 IPA 音标。它是整个服务的"理解层"：
     · 前端渲染靠它 —— 每个英文词的音标 ruby、逐句高亮，都基于这里输出的结构。
     · TTS/视频分句靠它 —— 一句一段音频、一句一帧画面。

 【为什么要自己切句/切词，不用现成分词器】
   中英混排 + 要区分"英文词 vs 中文字 vs 标点"这几类(渲染/注音各不同)，需求很具体；
   用一个精确的正则一次切开最简单可控，也不引入重依赖。

 【三个中文相关的坑】
   ① 句末标点中英不同：中文用 。！？，英文用 . ! ?，都要能断句(且英文句点要避开小数点/缩写误断)。
   ② 中文没有空格：不能按空格切词，所以中文按"单字"成 token(逐字可注音/逐字高亮)。
   ③ 英文才注音标：中文不注 IPA(注了没意义)，只有 kind=='en' 的 token 才查音标。

 【安全】所有上限(总字数/句数/单词长度)在这里校验，超限直接抛 ValidationError(见 core/exceptions)。
================================================================================
"""
import re

from ..core.config import get_settings
from ..core.exceptions import ValidationError
from .ipa import word_to_ipa

# 中文字符范围(用于识别中文字/切句)
_CJK = r"一-鿿㐀-䶿豈-﫿"
# 句末切分：中文标点整串、或"英文句点后跟空白/结尾"(避免把 3.14 / Mr. 误断)、或换行
_SENT_END = re.compile(rf"([。！？!?…]+|\.(?=\s|$)|\n+)")
# token 切分：英文词 | 数字 | 单个中文字 | 空白串 | 其它标点串
_TOKEN = re.compile(rf"([a-zA-Z][a-zA-Z']*|[0-9]+|[{_CJK}]|\s+|[^\sa-zA-Z0-9{_CJK}]+)")


def _classify(tok: str) -> str:
    """判断一个 token 属于哪类：en(英文词)/num(数字)/cjk(中文字)/space(空白)/punct(标点)。
    前端据此决定：en 上方注音标、cjk/punct 直接显示、space 保留间距。"""
    if re.fullmatch(r"[a-zA-Z][a-zA-Z']*", tok):
        return "en"
    if re.fullmatch(r"[0-9]+", tok):
        return "num"
    if re.fullmatch(rf"[{_CJK}]", tok):
        return "cjk"
    if tok.isspace():
        return "space"
    return "punct"


def split_sentences(text: str) -> list[str]:
    """按中英文句末标点/换行切句，并【保留标点】(朗读/显示都需要原样标点)。
    做法：用捕获组切分后逐段拼接，遇到"句末标记"就收束成一句。"""
    out, buf = [], ""
    for part in _SENT_END.split(text):        # split 保留分隔符(因为用了捕获组)
        if part is None:
            continue
        buf += part
        if _SENT_END.fullmatch(part):         # 这一段正好是句末标记 → 断句
            if buf.strip():
                out.append(buf.strip())
            buf = ""
    if buf.strip():                           # 收尾：最后没有标点的残句
        out.append(buf.strip())
    return out


def tokenize(sentence: str) -> list[dict]:
    """把一句切成 token 列表；英文词顺带查好 IPA(带来源标记：准/近似)。"""
    s = get_settings()
    tokens = []
    for m in _TOKEN.finditer(sentence):
        tok = m.group(0)
        kind = _classify(tok)
        item = {"text": tok, "kind": kind}
        if kind == "en":
            if len(tok) > s.max_word_len:                 # 异常超长"词"→ 拒绝(防脏输入)
                raise ValidationError(f"单词过长(>{s.max_word_len})。")
            ipa, src = word_to_ipa(tok)                   # 只有英文词才注音标
            item["ipa"], item["ipa_source"] = ipa, src
        tokens.append(item)
    return tokens


def analyze(text: str) -> dict:
    """入口：整段文本 → {sentences:[{index,text,tokens}], stats}。带完整输入校验。"""
    s = get_settings()
    text = (text or "").strip()
    if not text:
        raise ValidationError("文本不能为空。")
    if len(text) > s.max_text_chars:                      # 总长度上限
        raise ValidationError(f"文本过长(>{s.max_text_chars} 字符)。")
    sents = split_sentences(text)
    if len(sents) > s.max_sentences:                      # 句数上限
        raise ValidationError(f"句子过多(>{s.max_sentences})。")
    result = []
    approx = 0                                            # 统计有多少词用了"近似音标"(提示用户装 eng_to_ipa)
    for i, sent in enumerate(sents):
        toks = tokenize(sent)
        approx += sum(1 for t in toks if t.get("ipa_source") == "approx")
        result.append({"index": i, "text": sent, "tokens": toks})
    return {
        "sentences": result,
        "stats": {"sentences": len(result),
                  "en_words": sum(1 for se in result for t in se["tokens"] if t["kind"] == "en"),
                  "approx_ipa": approx},
    }
