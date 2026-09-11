"""规则检测:敏感词 + PII(手机号/邮箱)正则。离线、确定。生产可扩违规词库/模型。"""
import re
SENSITIVE = ["暴力", "诈骗", "赌博", "fuck", "违禁"]
_PHONE = re.compile(r"1[3-9]\d{9}")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def find_sensitive(text):
    return [w for w in SENSITIVE if w in text.lower()]


def find_pii(text):
    return {"phone": _PHONE.findall(text), "email": _EMAIL.findall(text)}


def mask_pii(text):
    text = _PHONE.sub(lambda m: m.group()[:3] + "****" + m.group()[-2:], text)
    text = _EMAIL.sub(lambda m: m.group()[0] + "***@***", text)
    return text
