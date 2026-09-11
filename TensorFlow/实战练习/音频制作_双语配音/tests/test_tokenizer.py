import pytest

from app.core.exceptions import ValidationError
from app.services.tokenizer import analyze, split_sentences


def test_split_mixed_sentences():
    s = split_sentences("Hello 你好。Today is good! 再见")
    assert len(s) >= 2


def test_analyze_tags_english_ipa():
    r = analyze("Hello 你好")
    toks = r["sentences"][0]["tokens"]
    en = [t for t in toks if t["kind"] == "en"]
    cjk = [t for t in toks if t["kind"] == "cjk"]
    assert en and en[0]["ipa"] and cjk
    assert r["stats"]["en_words"] == 1


def test_empty_rejected():
    with pytest.raises(ValidationError):
        analyze("   ")
