"""
本文件是《从零构建大语言模型》(Build a Large Language Model From Scratch) 一书中
第 2 章「文本数据处理」(BPE 分词器) 部分的 **测试文件**。

背景说明：
- 第 2 章介绍了 BPE (Byte Pair Encoding，字节对编码) 分词算法，并在配套的 Jupyter Notebook
  文件 `bpe-from-scratch.ipynb` 中实现了一个教学用的分词器类 `BPETokenizerSimple`，
  以及一个辅助函数 `download_file_if_absent`（用于按需下载训练/测试所需的数据文件）。
- 由于教学代码写在 `.ipynb` 笔记本里而不是普通的 `.py` 模块中，本文件通过
  `import_definitions_from_notebook` 这个小工具函数，把 notebook 中的代码单元格
  当作模块代码执行一遍，从而把 `BPETokenizerSimple`、`download_file_if_absent`
  等定义“提取”出来，供 pytest 测试用例导入和调用。
- 测试内容主要包括：
  1. 从零训练一个简易 BPE 分词器（在《The Verdict》短篇小说文本上训练），验证词表大小、
     合并规则数量、编码/解码结果，以及词表与合并规则的保存与重新加载是否正确；
  2. 加载 OpenAI 官方发布的 GPT-2 词表和合并规则文件，验证自制分词器与
     OpenAI 官方分词器 (通过 `tiktoken` 库) 在各种边界情况（换行符、多个空格、
     特殊标记 `<|endoftext|>` 等）下的编码/解码结果是否一致。

本文件不涉及神经网络的前向传播或张量运算，测试对象是「文本 -> token id 序列」以及
「token id 序列 -> 文本」这两个方向的正确性，因此代码注释中不会出现张量形状说明，
而是着重解释分词逻辑、边界情况和测试断言的含义。
"""
import os
import sys
import io
import nbformat
import types
import pytest

import tiktoken


def import_definitions_from_notebook(fullname, names):
    """Loads function definitions from a Jupyter notebook file into a module.

    中文说明：
        本函数的作用是把一个 `.ipynb` Jupyter Notebook 文件当作一个 Python 模块来“导入”。
        因为教学代码（比如 `BPETokenizerSimple` 类和 `download_file_if_absent` 函数）
        写在 notebook 的代码单元格里，而不是普通的 `.py` 文件里，pytest 没办法直接
        `import` 它们，所以需要手动读取 notebook 文件、按顺序执行其中的每一个代码单元格，
        把执行后产生的变量/函数/类都收集到一个新建的模块对象中。

    参数:
        fullname (str): notebook 文件的“模块名”，同时也是文件名（不含 .ipynb 后缀），
            例如 "bpe-from-scratch" 对应磁盘上的 "bpe-from-scratch.ipynb"。
        names (list[str]): 期望从 notebook 中提取出来的顶层名称列表（函数名/类名等），
            用于校验 notebook 是否包含了测试所需要的全部定义。

    返回值:
        types.ModuleType: 一个动态创建的模块对象，其 `__dict__` 中包含了 notebook 里
            所有代码单元格执行后产生的全局变量，包括 `names` 中要求的那些定义。

    异常:
        FileNotFoundError: 如果对应的 .ipynb 文件不存在。
        ImportError: 如果执行完所有代码单元格后，`names` 中列出的某些名称仍然缺失。
    """
    # 拼出 notebook 文件的绝对路径：与本测试文件同目录，文件名为 fullname + ".ipynb"
    path = os.path.join(os.path.dirname(__file__), fullname + ".ipynb")
    path = os.path.normpath(path)  # 规范化路径（处理 ../ 之类的相对路径写法）

    if not os.path.exists(path):
        raise FileNotFoundError(f"Notebook file not found at: {path}")

    # 以 nbformat 库解析 notebook 的 JSON 结构（notebook 文件本质上是 JSON）
    with io.open(path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)  # as_version=4 表示按 nbformat v4 规范解析

    # 动态创建一个空模块，并注册到 sys.modules，
    # 这样后续 exec 执行 notebook 代码时，函数/类内部的全局命名空间就是这个模块的 __dict__，
    # 效果等价于把 notebook 内容当成一个真正的 .py 模块导入。
    mod = types.ModuleType(fullname)
    sys.modules[fullname] = mod

    # Execute all code cells to capture dependencies
    # 中文：依次执行 notebook 中所有「代码类型」的单元格（跳过 markdown 说明单元格），
    # exec 的第二个参数 mod.__dict__ 作为全局命名空间，
    # 这样单元格里定义的函数、类、变量都会被写入 mod 模块对象中。
    for cell in nb.cells:
        if cell.cell_type == "code":
            exec(cell.source, mod.__dict__)

    # Ensure required names are in module
    # 中文：执行完所有代码单元格后，检查调用方要求的 names 是否都已经出现在模块命名空间里，
    # 如果 notebook 内容发生变化（比如函数改名了），这里会及时报错，方便定位问题。
    missing_names = [name for name in names if name not in mod.__dict__]
    if missing_names:
        raise ImportError(f"Missing definitions in notebook: {missing_names}")

    return mod


@pytest.fixture(scope="module")
def imported_module():
    """pytest fixture：把 bpe-from-scratch.ipynb 中的定义加载为一个 Python 模块。

    scope="module" 表示这个 fixture 在同一个测试模块（本文件）内只会执行一次并被缓存，
    所有测试函数共用同一份加载结果，避免重复解析/执行 notebook 带来的额外开销。

    返回值:
        types.ModuleType: 包含 `BPETokenizerSimple` 类和 `download_file_if_absent`
            函数的模块对象。
    """
    fullname = "bpe-from-scratch"
    names = ["BPETokenizerSimple", "download_file_if_absent"]
    return import_definitions_from_notebook(fullname, names)


@pytest.fixture(scope="module")
def verdict_file(imported_module):
    """Fixture to handle downloading The Verdict file.

    中文说明：
        提供训练用的原始文本文件《The Verdict》（第 2 章从头到尾使用的示例短篇小说），
        如果本地尚未下载过，则会从 GitHub 上按需下载；如果已经存在（比如第 2 章主代码
        已经下载过一份），则直接复用，避免重复下载。

    参数:
        imported_module: 上面的 `imported_module` fixture，提供 `download_file_if_absent`。

    返回值:
        str: 本地磁盘上《The Verdict》文本文件的路径。
    """
    download_file_if_absent = getattr(imported_module, "download_file_if_absent", None)

    verdict_path = download_file_if_absent(
        url=(
            "https://raw.githubusercontent.com/rasbt/"
            "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
            "the-verdict.txt"
        ),
        filename="the-verdict.txt",
        # 依次在这些目录中查找是否已存在该文件，命中则不重新下载
        search_dirs=["ch02/01_main-chapter-code/", "../01_main-chapter-code/", "."]
    )

    return verdict_path


@pytest.fixture(scope="module")
def gpt2_files(imported_module):
    """Fixture to handle downloading GPT-2 files.

    中文说明：
        下载（或复用本地已有的）OpenAI 官方发布的 GPT-2 分词器所需的两个文件：
        - encoder.json：GPT-2 的词表文件（token 字符串 <-> id 的映射表，JSON 格式）；
        - vocab.bpe：GPT-2 的 BPE 合并规则文件（记录了训练时学到的字节对合并顺序）。
        有了这两个文件，就可以在本地精确重建 OpenAI 官方的 GPT-2 分词器行为，
        用来跟自制的 `BPETokenizerSimple` 做一致性比对测试。

    返回值:
        dict[str, str]: 形如 {"vocab.bpe": 本地路径, "encoder.json": 本地路径} 的映射。
    """
    download_file_if_absent = getattr(imported_module, "download_file_if_absent", None)

    search_directories = ["ch02/02_bonus_bytepair-encoder/gpt2_model/", "../02_bonus_bytepair-encoder/gpt2_model/", "."]
    files_to_download = {
        "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe": "vocab.bpe",
        "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json": "encoder.json"
    }
    # 对每个 (url, filename) 依次下载（或从本地缓存目录中找到已存在的文件），
    # 结果收集为一个以文件名为 key 的字典，方便测试用例通过文件名取用路径
    paths = {filename: download_file_if_absent(url, filename, search_directories)
             for url, filename in files_to_download.items()}

    return paths


def test_tokenizer_training(imported_module, verdict_file):
    """测试：从零在《The Verdict》文本上训练一个 BPE 分词器，并验证训练/编码/解码/持久化的正确性。

    覆盖的关键点：
    1. 训练后词表大小、合并规则数量是否符合预期（回归测试，锁定算法实现的确定性输出）；
    2. 训练过程中不应该产生「非法」的带空格标记（Ġ 出现在非开头位置）——
       Ġ 是 GPT-2 风格分词法里用来表示“该 token 前面有一个空格”的特殊字符，
       正常情况下 Ġ 只应出现在 token 的最前面；
    3. 对一句英文的编码结果是否与预期 token id 序列完全一致；
    4. 编码后再解码，应能还原出原始输入文本（无损往返，round-trip）；
    5. 把词表和合并规则保存到磁盘文件，再用一个新的分词器实例重新加载，
       解码结果应与原分词器一致，验证序列化/反序列化的正确性。
    """
    BPETokenizerSimple = getattr(imported_module, "BPETokenizerSimple", None)
    tokenizer = BPETokenizerSimple()

    with open(verdict_file, "r", encoding="utf-8") as f:  # added ../01_main-chapter-code/
        text = f.read()

    # 在原始文本上训练 BPE：vocab_size=1000 表示训练到词表达到 1000 个 token 为止，
    # allowed_special 指定训练语料中允许出现的特殊标记（这里是文本结束符 <|endoftext|>）
    tokenizer.train(text, vocab_size=1000, allowed_special={"<|endoftext|>"})
    assert len(tokenizer.vocab) == 1000, "Tokenizer vocabulary size mismatch."
    # BPE 合并规则数 = 最终词表大小 - 初始词表大小（初始词表通常是 256 个字节 + 特殊标记等），
    # 这里固定断言为 742，用来确保算法实现没有被意外改动
    assert len(tokenizer.bpe_merges) == 742, "Tokenizer BPE merges count mismatch."

    input_text = "Jack embraced beauty through art and life."
    # 检查词表中是否存在「非法」的 Ġ 用法：
    # Ġ 只应该出现在 token 的最前面（表示该 token 前面跟着一个空格），
    # 如果 Ġ 出现在字符串中间或末尾，说明训练/合并逻辑有 bug
    invalid_whitespace_tokens = [
        tok for tok in tokenizer.vocab.values()
        if "Ġ" in tok and tok != "Ġ" and not tok.startswith("Ġ")
    ]
    assert not invalid_whitespace_tokens, "Training should not learn tokens with non-leading Ġ markers."

    token_ids = tokenizer.encode(input_text)
    # 精确比对编码结果，属于回归测试：只要训练语料、训练超参数、算法实现都不变，
    # 编码结果应当是完全确定的
    assert token_ids == [74, 361, 310, 109, 98, 420, 397, 100, 300, 428, 116, 121, 519, 699, 299, 808, 534], "Token IDs do not match expected output."

    # 编码后再解码，验证「文本 -> id -> 文本」的无损往返
    assert tokenizer.decode(token_ids) == input_text, "Decoded text does not match the original input."

    # 将当前训练好的词表与合并规则保存到磁盘（分别是 JSON 格式的词表和文本格式的合并规则）
    tokenizer.save_vocab_and_merges(vocab_path="vocab.json", bpe_merges_path="bpe_merges.txt")
    # 用一个全新的分词器实例，从刚保存的文件重新加载词表和合并规则
    tokenizer2 = BPETokenizerSimple()
    tokenizer2.load_vocab_and_merges(vocab_path="vocab.json", bpe_merges_path="bpe_merges.txt")
    # 重新加载后的分词器应当能正确解码同一组 token_ids，验证序列化/反序列化没有丢失信息
    assert tokenizer2.decode(token_ids) == input_text, "Decoded text mismatch after reloading tokenizer."


def test_gpt2_tokenizer_openai_simple(imported_module, gpt2_files):
    """测试：加载 OpenAI 官方 GPT-2 的词表/合并规则文件后，基本编码功能是否正确。

    通过 `load_vocab_and_merges_from_openai` 直接复用 OpenAI 发布的 encoder.json 和
    vocab.bpe 文件（而不是自己训练），从而让自制分词器具备与官方 GPT-2 完全一致的
    词表（50257 个 token，包含字节级 BPE 基础词表 + 常见合并 + 特殊标记）。
    """
    BPETokenizerSimple = getattr(imported_module, "BPETokenizerSimple", None)

    tokenizer_gpt2 = BPETokenizerSimple()
    tokenizer_gpt2.load_vocab_and_merges_from_openai(
        vocab_path=gpt2_files["encoder.json"], bpe_merges_path=gpt2_files["vocab.bpe"]
    )

    # GPT-2（124M 版本）官方词表大小固定为 50257
    assert len(tokenizer_gpt2.vocab) == 50257, "GPT-2 tokenizer vocabulary size mismatch."

    input_text = "This is some text"
    token_ids = tokenizer_gpt2.encode(input_text)
    # 与 OpenAI 官方分词结果对照的固定预期值（回归测试）
    assert token_ids == [1212, 318, 617, 2420], "Tokenized output does not match expected GPT-2 encoding."


def test_gpt2_tokenizer_openai_edgecases(imported_module, gpt2_files):
    """测试：在多种边界情况文本上，自制分词器与 tiktoken（OpenAI 官方分词库）的编码结果是否一致。

    这里同时校验了两件事：
    1. `tiktoken.get_encoding("gpt2")` 的编码结果是否与我们手写的「预期值」一致
       （用于确认预期值本身没写错，是可信的黄金标准）；
    2. 自制的 `BPETokenizerSimple`（加载了 OpenAI 官方词表/合并规则）编码结果
       是否也与该预期值一致（用于验证自制实现与官方实现行为等价）。

    测试用例覆盖了：结尾逗号、含多个子词合并的长单词、含大量标点/数字混杂的怪异字符串，
    以及包含双连字符 "--" 的常见英文句子等场景。
    """
    BPETokenizerSimple = getattr(imported_module, "BPETokenizerSimple", None)

    tokenizer_gpt2 = BPETokenizerSimple()
    tokenizer_gpt2.load_vocab_and_merges_from_openai(
        vocab_path=gpt2_files["encoder.json"], bpe_merges_path=gpt2_files["vocab.bpe"]
    )
    # tiktoken 是 OpenAI 官方发布的高性能分词库，这里作为「标准答案」来源之一
    tik_tokenizer = tiktoken.get_encoding("gpt2")

    test_cases = [
        ("Hello,", [15496, 11]),
        ("Implementations", [3546, 26908, 602]),
        ("asdf asdfasdf a!!, @aba 9asdf90asdfk", [292, 7568, 355, 7568, 292, 7568, 257, 3228, 11, 2488, 15498, 860, 292, 7568, 3829, 292, 7568, 74]),
        ("Hello, world. Is this-- a test?", [15496, 11, 995, 13, 1148, 428, 438, 257, 1332, 30])
    ]

    errors = []  # 收集所有失败信息，最后一次性汇报，方便定位所有不一致的用例而不是遇到第一个就停止

    for input_text, expected_tokens in test_cases:
        tik_tokens = tik_tokenizer.encode(input_text)
        gpt2_tokens = tokenizer_gpt2.encode(input_text)

        # 打印调试信息，方便测试失败时在终端里对照排查具体是哪个环节出现差异
        print(f"Text: {input_text}")
        print(f"Expected Tokens: {expected_tokens}")
        print(f"tiktoken Output: {tik_tokens}")
        print(f"BPETokenizerSimple Output: {gpt2_tokens}")
        print("-" * 40)

        if tik_tokens != expected_tokens:
            errors.append(f"Tiktokenized output does not match expected GPT-2 encoding for '{input_text}'.\n"
                          f"Expected: {expected_tokens}, Got: {tik_tokens}")

        if gpt2_tokens != expected_tokens:
            errors.append(f"Tokenized output does not match expected GPT-2 encoding for '{input_text}'.\n"
                          f"Expected: {expected_tokens}, Got: {gpt2_tokens}")

    if errors:
        # 把所有收集到的错误信息合并成一条失败消息一起抛出
        pytest.fail("\n".join(errors))


def test_gpt2_newline_and_eot_ids(imported_module, gpt2_files):
    """测试：GPT-2 词表中换行符与文本结束符 (<|endoftext|>) 的 id 映射是否正确。

    背景知识：
    - GPT-2 使用字节级 BPE，为了让空白字符也能参与 BPE 合并，会把常见的空白字符
      用不可见的占位符号表示，比如换行符 "\\n" 在词表里对应的字符是 "Ċ"（Unicode 中的
      一个特殊符号），空格对应 "Ġ"。
    - "<|endoftext|>" 是 GPT-2 用来标记文档结尾的特殊 token，在官方词表中固定映射到 id 50256
      （即词表中的最后一个 id，因为基础词表是 0~50255，特殊标记追加在最后）。
    """
    BPETokenizerSimple = getattr(imported_module, "BPETokenizerSimple", None)

    tok = BPETokenizerSimple()
    tok.load_vocab_and_merges_from_openai(
        vocab_path=gpt2_files["encoder.json"], bpe_merges_path=gpt2_files["vocab.bpe"]
    )

    # inverse_vocab：token 字符串 -> id 的反向映射表，用于校验特定符号是否存在
    assert "Ċ" in tok.inverse_vocab, "Missing GPT-2 newline glyph 'Ċ' in inverse_vocab"
    assert "<|endoftext|>" in tok.inverse_vocab, "Missing EOT in inverse_vocab"

    # "Ċ"（换行符的 GPT-2 编码表示）固定对应 id 198
    assert tok.inverse_vocab["Ċ"] == 198, "Ċ must map to id 198"
    # "<|endoftext|>" 固定对应 id 50256（GPT-2 词表末尾的特殊标记）
    assert tok.inverse_vocab["<|endoftext|>"] == 50256, "EOT must be 50256"

    # 如果反向词表中还没有真实的换行符 "\n" 这个 key（有些实现只存了占位符 "Ċ"），
    # 就手动补上一条 "\n" -> 198 的映射，保证后续可以用真正的换行符查表
    if "\n" not in tok.inverse_vocab:
        tok.inverse_vocab["\n"] = tok.inverse_vocab["Ċ"]
    assert tok.inverse_vocab["\n"] == 198, r"'\n' must map to 198 via Ċ"

    # 正向词表 vocab[id] -> token 字符串 也要保持一致：不能因为上面补充了 "\n" 的映射
    # 而破坏了 id 198 原本对应的字符串（应仍然是占位符 "Ċ"，而不是被误改成别的东西）
    assert tok.vocab[198] == "Ċ", "Don't overwrite vocab[198]; keep it 'Ċ'"
    assert tok.vocab[50256] == "<|endoftext|>", "Don't map <|endoftext|> to anything else"


def test_no_eot_aliasing_and_disallowed_logic(imported_module, gpt2_files):
    """测试：默认情况下遇到特殊标记应报错，显式允许后编码结果应与 tiktoken 一致。

    这是在验证「特殊标记的安全机制」：像 <|endoftext|> 这种特殊 token 通常不应该
    随随便便出现在用户输入的普通文本里（否则可能被恶意注入，冒充文档边界），
    所以像 tiktoken 一样，`encode` 方法在默认情况下（未显式声明 allowed_special）
    遇到特殊标记文本应当直接抛出 ValueError，只有调用方显式传入
    `allowed_special={"<|endoftext|>"}` 时才允许正常编码。
    """
    BPETokenizerSimple = getattr(imported_module, "BPETokenizerSimple", None)
    tok = BPETokenizerSimple()
    tok.load_vocab_and_merges_from_openai(
        vocab_path=gpt2_files["encoder.json"], bpe_merges_path=gpt2_files["vocab.bpe"]
    )
    tik = tiktoken.get_encoding("gpt2")

    text = "Hello<|endoftext|>\nworld"
    # When not allowed, our encode should raise ValueError like tiktoken
    # 中文：默认（未声明 allowed_special）情况下，文本中出现 <|endoftext|> 应该报错，
    # 这与 tiktoken 的安全默认行为保持一致，防止特殊标记被意外/恶意注入
    with pytest.raises(ValueError):
        tok.encode(text)

    # When allowed, both tokenizers should match
    # 中文：显式声明允许 <|endoftext|> 出现后，两个分词器的编码结果应当完全一致
    ids_ours = tok.encode(text, allowed_special={"<|endoftext|>"})
    ids_tik = tik.encode(text, allowed_special={"<|endoftext|>"})
    assert ids_ours == ids_tik, "Mismatch vs tiktoken with EOT allowed"


@pytest.mark.parametrize(
    "text",
    [
        "a\nb",
        "a\n\nb",
        "\nHello",
        "Hello\n",
        "a\r\nb",
    ],
)
def test_newline_roundtrip_and_equivalence(imported_module, gpt2_files, text):
    """参数化测试：多种含换行符（含 \\r\\n）的文本，验证编码一致性与「编码后解码可还原」。

    使用 `@pytest.mark.parametrize` 让 pytest 对列表中的每一个 `text` 分别执行一次本测试函数，
    覆盖单个换行、连续两个换行（空行）、开头换行、结尾换行，以及 Windows 风格的 "\\r\\n" 换行符等场景。
    """
    BPETokenizerSimple = getattr(imported_module, "BPETokenizerSimple", None)
    tok = BPETokenizerSimple()
    tok.load_vocab_and_merges_from_openai(
        vocab_path=gpt2_files["encoder.json"], bpe_merges_path=gpt2_files["vocab.bpe"]
    )
    tik = tiktoken.get_encoding("gpt2")

    ids_ours = tok.encode(text)
    ids_tik = tik.encode(text)

    # 自制分词器与 tiktoken 对同一段含换行符的文本编码结果应完全相同
    assert ids_ours == ids_tik, f"Mismatch vs tiktoken for: {repr(text)}"
    # Each "\n" should correspond to id 198
    # 中文：文本中每出现一个 "\n"，编码结果里就应该恰好多出现一次 id 198（换行符对应的 token id），
    # 这是对换行符编码规则的一个额外、更细粒度的交叉验证
    expected_lf_count = text.count("\n")
    assert ids_ours.count(198) == expected_lf_count

    # 验证「编码 -> 解码」的无损往返：解码结果应与原始输入文本完全一致
    dec = tok.decode(ids_ours)
    assert dec == text


def test_space_newline_space_patterns(imported_module, gpt2_files):
    """测试：空格与换行符相邻出现（空格+换行 / 换行+空格）时的编码是否与 tiktoken 一致。

    这类场景容易暴露分词器在「合并空白字符」逻辑上的边界 bug，比如是否错误地把
    换行符前/后的空格吞并进了错误的 token 里。
    """
    BPETokenizerSimple = getattr(imported_module, "BPETokenizerSimple", None)
    tok = BPETokenizerSimple()
    tok.load_vocab_and_merges_from_openai(
        vocab_path=gpt2_files["encoder.json"], bpe_merges_path=gpt2_files["vocab.bpe"]
    )
    tik = tiktoken.get_encoding("gpt2")

    samples = [
        "Hello \nworld",   # 单词后接空格，再接换行符
        "Hello\n world",   # 单词后接换行符，再接空格
    ]
    for s in samples:
        assert tok.encode(s) == tik.encode(s), f"Mismatch vs tiktoken: {repr(s)}"


def test_multiple_leading_spaces_roundtrip(imported_module, gpt2_files):
    """测试：文本开头包含多个连续空格时，编码后再解码是否能完整还原原始文本。

    多个连续空格在字节级 BPE 中比较特殊（GPT-2 词表里专门有代表「多个连续空格」的
    合并 token），这里只关注「编码 -> 解码」的往返是否无损，不与 tiktoken 做交叉对比。
    """
    BPETokenizerSimple = getattr(imported_module, "BPETokenizerSimple", None)
    tok = BPETokenizerSimple()
    tok.load_vocab_and_merges_from_openai(
        vocab_path=gpt2_files["encoder.json"], bpe_merges_path=gpt2_files["vocab.bpe"]
    )

    text = "  Hello World."
    # 编码后立即解码，结果应与原始文本一模一样（包括开头的两个空格）
    assert tok.decode(tok.encode(text)) == text
