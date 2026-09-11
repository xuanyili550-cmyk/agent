"""
================================================================================
 服务层 · 英文 → IPA 音标（三级来源：库 → 内置词典 → 规则兜底）
================================================================================
 【这个文件做什么】给一个英文单词返回它的国际音标(IPA)，供前端在词上方注音、视频里标音标。

 【为什么是"三级降级"设计(核心)】
   准确的英文注音需要一份发音词典(几万词)，没有它就只能靠拼写规则猜(英文拼写≠发音，很不准)。
   所以按"能拿到多准就用多准"分三级：
     ① eng_to_ipa 库(装了就用) —— 内含 CMU 发音词典，全量准确。pip install eng-to-ipa 即自动启用。
     ② 内置常用词词典 BUILTIN —— 没装库时，高频词(问候/常用动词名词/功能词)仍然准。
     ③ 规则兜底 _rough —— 生僻词用字母/字母组合近似，标记为 approx(前端会提示"近似，不保证准")。
   这样：装了库=全准；没装=常用词准、生僻词近似 —— 优雅降级，不因缺依赖就完全没音标。

 【性能】word_to_ipa 带 LRU 缓存：同一个词(如 the/you)在一篇里出现很多次，只查一次。
 【中文】不注音标——只处理英文词。
================================================================================
"""
import re
from functools import lru_cache

# ② 内置常用词 IPA 词典(高频词，够学习卡/课文场景；生僻词建议装 eng-to-ipa 走全量)
BUILTIN = {
    "hello": "həˈloʊ", "hi": "haɪ", "hey": "heɪ", "world": "wɜːrld", "good": "ɡʊd",
    "morning": "ˈmɔːrnɪŋ", "afternoon": "ˌæftərˈnuːn", "evening": "ˈiːvnɪŋ", "night": "naɪt",
    "thank": "θæŋk", "thanks": "θæŋks", "you": "juː", "please": "pliːz", "sorry": "ˈsɑːri",
    "yes": "jes", "no": "noʊ", "okay": "ˌoʊˈkeɪ", "ok": "ˌoʊˈkeɪ", "bye": "baɪ",
    "welcome": "ˈwelkəm", "name": "neɪm", "my": "maɪ", "your": "jɔːr", "is": "ɪz", "am": "æm",
    "are": "ɑːr", "the": "ðə", "a": "ə", "an": "æn", "and": "ænd", "or": "ɔːr", "but": "bʌt",
    "i": "aɪ", "he": "hiː", "she": "ʃiː", "it": "ɪt", "we": "wiː", "they": "ðeɪ",
    "this": "ðɪs", "that": "ðæt", "these": "ðiːz", "those": "ðoʊz", "here": "hɪr", "there": "ðer",
    "what": "wʌt", "who": "huː", "where": "wer", "when": "wen", "why": "waɪ", "how": "haʊ",
    "which": "wɪtʃ", "can": "kæn", "will": "wɪl", "would": "wʊd", "should": "ʃʊd", "could": "kʊd",
    "do": "duː", "does": "dʌz", "did": "dɪd", "have": "hæv", "has": "hæz", "had": "hæd",
    "go": "ɡoʊ", "going": "ˈɡoʊɪŋ", "come": "kʌm", "get": "ɡet", "make": "meɪk", "made": "meɪd",
    "see": "siː", "look": "lʊk", "want": "wɑːnt", "need": "niːd", "like": "laɪk", "love": "lʌv",
    "know": "noʊ", "think": "θɪŋk", "say": "seɪ", "said": "sed", "tell": "tel", "ask": "æsk",
    "work": "wɜːrk", "study": "ˈstʌdi", "learn": "lɜːrn", "read": "riːd", "write": "raɪt",
    "speak": "spiːk", "listen": "ˈlɪsən", "eat": "iːt", "drink": "drɪŋk", "water": "ˈwɔːtər",
    "food": "fuːd", "apple": "ˈæpəl", "banana": "bəˈnænə", "book": "bʊk", "school": "skuːl",
    "teacher": "ˈtiːtʃər", "student": "ˈstuːdənt", "friend": "frend", "family": "ˈfæməli",
    "time": "taɪm", "day": "deɪ", "today": "təˈdeɪ", "tomorrow": "təˈmɑːroʊ", "year": "jɪr",
    "one": "wʌn", "two": "tuː", "three": "θriː", "four": "fɔːr", "five": "faɪv",
    "computer": "kəmˈpjuːtər", "phone": "foʊn", "internet": "ˈɪntərnet", "language": "ˈlæŋɡwɪdʒ",
    "english": "ˈɪŋɡlɪʃ", "chinese": "tʃaɪˈniːz", "word": "wɜːrd", "sentence": "ˈsentəns",
    "beautiful": "ˈbjuːtɪfəl", "happy": "ˈhæpi", "big": "bɪɡ", "small": "smɔːl", "new": "nuː",
    "old": "oʊld", "run": "rʌn", "walk": "wɔːk", "play": "pleɪ", "music": "ˈmjuːzɪk",
    "video": "ˈvɪdioʊ", "audio": "ˈɔːdioʊ", "voice": "vɔɪs", "sound": "saʊnd", "for": "fɔːr",
    "to": "tuː", "of": "ʌv", "in": "ɪn", "on": "ɑːn", "with": "wɪð", "at": "æt", "from": "frʌm",
}

# ③ 兜底：字母/字母组合 → 近似音。★顺序重要：长的组合(tion/sh/ee)必须排在单字母前，先匹配长的
_RULES = [("tion", "ʃən"), ("sh", "ʃ"), ("ch", "tʃ"), ("th", "θ"), ("ph", "f"), ("ck", "k"),
          ("ee", "iː"), ("oo", "uː"), ("ou", "aʊ"), ("ai", "eɪ"), ("ay", "eɪ"), ("ea", "iː"),
          ("a", "æ"), ("e", "e"), ("i", "ɪ"), ("o", "ɑː"), ("u", "ʌ"), ("y", "i"),
          ("c", "k"), ("g", "ɡ"), ("j", "dʒ"), ("q", "k"), ("x", "ks"), ("r", "r")]

_ENG = None   # eng_to_ipa 模块的缓存句柄(None=还没探测, False=没装, 模块=装了)


def _lib():
    """探测一次 eng_to_ipa 是否可用并缓存结果——避免每次调用都 try import。"""
    global _ENG
    if _ENG is None:
        try:
            import eng_to_ipa as e
            _ENG = e
        except Exception:
            _ENG = False
    return _ENG


def _rough(w: str) -> str:
    """③ 规则兜底：从左到右贪心匹配 _RULES(长组合优先)，拼出近似音。仅供生僻词，不保证准。"""
    out, i = "", 0
    while i < len(w):
        for pat, rep in _RULES:
            if w[i:i + len(pat)] == pat:
                out += rep
                i += len(pat)
                break
        else:                     # 没有任何规则匹配 → 原样保留该字母
            out += w[i]
            i += 1
    return out


@lru_cache(maxsize=8192)
def word_to_ipa(word: str) -> tuple[str, str]:
    """返回 (ipa含斜杠, 来源)。来源: eng_to_ipa|builtin|approx|none。带 LRU 缓存(热词零成本)。"""
    w = re.sub(r"[^a-zA-Z']", "", word).lower()      # 去掉标点/数字，统一小写再查
    if not w:
        return "", "none"
    lib = _lib()
    if lib:                                          # ① 有库：全量准确
        try:
            ipa = lib.convert(w)
            if ipa and "*" not in ipa:               # eng_to_ipa 用 * 标"查不到"，带 * 说明它也没有
                return f"/{ipa}/", "eng_to_ipa"
        except Exception:
            pass                                     # 库出错也不能让整个请求挂 → 继续降级
    if w in BUILTIN:                                 # ② 内置词典：常用词准
        return f"/{BUILTIN[w]}/", "builtin"
    return f"/{_rough(w)}/", "approx"                # ③ 规则兜底：近似(前端标注)
