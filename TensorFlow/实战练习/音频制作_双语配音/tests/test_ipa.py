from app.services.ipa import word_to_ipa


def test_builtin_known_word():
    ipa, src = word_to_ipa("Hello")
    assert ipa.startswith("/") and ipa.endswith("/")
    assert src in ("eng_to_ipa", "builtin")


def test_punctuation_stripped():
    ipa, _ = word_to_ipa("world,")
    assert "," not in ipa


def test_unknown_word_has_fallback():
    ipa, src = word_to_ipa("zxqwv")
    assert ipa and src in ("approx", "eng_to_ipa")


def test_empty():
    assert word_to_ipa("123")[1] == "none"
