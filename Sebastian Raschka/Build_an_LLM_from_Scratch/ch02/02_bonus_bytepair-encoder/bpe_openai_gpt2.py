# Source: https://github.com/openai/gpt-2/blob/master/src/encoder.py
# License:
# Modified MIT License

# Software Copyright (c) 2019 OpenAI

# We don’t claim ownership of the content you create with GPT-2, so it is yours to do with as you please.
# We only ask that you use GPT-2 responsibly and clearly indicate your content was created using GPT-2.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# The above copyright notice and this permission notice need not be included
# with content created by the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.
"""
【中文说明】本文件是直接照搬自 OpenAI 官方 GPT-2 仓库(gpt-2/src/encoder.py)的
字节对编码(Byte Pair Encoding, BPE)分词器实现,未修改任何逻辑。

在《从零构建大语言模型》(Build a Large Language Model From Scratch)一书中,
第 2 章(数据准备与采样)会介绍如何把原始文本切分成 token id 序列,以便
输入到 Transformer 模型中。本文件属于该章的"补充材料"(02_bonus_bytepair-encoder),
用途是:
    1. 提供与 OpenAI 官方 GPT-2/GPT-3 完全一致的 BPE 分词器实现,
       方便读者对比自己(或使用 tiktoken 库)实现的 BPE 分词结果是否正确;
    2. 演示 BPE 算法的核心步骤——
       (a) 先把文本按 Unicode 正则切分成"预分词"(pre-tokenize)片段;
       (b) 把每个片段的原始字节(byte)映射为可打印的 Unicode 字符,避免出现
           控制字符/不可见字符导致的编码问题;
       (c) 在每个片段内部反复合并"当前排名最靠前(即在训练语料中出现频率最高)"
           的相邻符号对,直到无法再合并,得到若干 BPE 子词(subword);
       (d) 通过词表(encoder.json)把子词字符串映射为整数 token id。
    3. 提供从 OpenAI 官方服务器下载 GPT-2 词表文件(encoder.json、vocab.bpe)的
       辅助函数 download_vocab,以及根据本地词表文件构建 Encoder 对象的
       get_encoder 函数。

注意:本文件只新增了中文注释,未改动任何原始代码逻辑、变量名或结构。
"""
import  os
import json
import regex as re  # 注意:这里用的是第三方 regex 库(而非标准库 re),
                     # 因为下面的正则表达式用到了 \p{L}、\p{N} 这类 Unicode 属性
                     # 转义,标准库 re 不支持,必须依赖 regex 库
import requests
from tqdm import tqdm  # 用于下载词表文件时显示进度条
from functools import lru_cache  # 用于缓存 bytes_to_unicode() 的计算结果,避免重复构建映射表

@lru_cache()  # 该函数的返回值与输入无关(无参数),加缓存后多次调用只计算一次,提升效率
def bytes_to_unicode():
    """
       Returns list of utf-8 byte and a corresponding list of unicode strings.
       The reversible bpe codes work on unicode strings.
       This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
       When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
       This is a significant percentage of your normal, say, 32K bpe vocab.
       To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
       And avoids mapping to whitespace/control characters the bpe code barfs on.

       【中文说明】
       作用:构建一个"字节 -> 可打印 Unicode 字符"的双射(一一对应)映射表。
       原因:UTF-8 编码后的原始字节(0~255)中,有很多是空白符、控制字符等
       不可打印/不可见字符,如果直接把这些字节值当作 BPE 算法处理的"符号"，
       会导致词表里出现无法正常显示、也容易在字符串处理中出错的字符。
       因此这里把 256 个字节值一一映射到"看得见"的 Unicode 码位上，
       这样 BPE 的合并操作就可以安全地在字符串层面进行，同时保证编码可逆
       (即可以无损地从 BPE 结果还原回原始字节)。
       参数:无
       返回值:dict,键为 0~255 的整数字节值，值为对应的单个 Unicode 字符（长度为 1 的字符串）。
       """
    # 第一步:先收集"本身就是可打印字符"的字节区间——
    # ord("!")~ord("~") 是 ASCII 可打印字符(33~126)，
    # ord("¡")~ord("¬")、ord("®")~ord("ÿ") 是 Latin-1 补充字符集中可打印的部分，
    # 这些字节值可以直接复用自身对应的 Unicode 码位，不需要额外映射
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]  # cs 初始时与 bs 相同(可打印字节直接映射到自身码位)
    n = 0
    # 第二步:遍历所有 256 个可能的字节值(0~255)，
    # 把不在上面"可打印集合"里的字节(例如空格、换行、控制字符等)
    # 映射到 256 以后的一段"私有安全区"码位(不会与常见可打印字符冲突)
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)  # 依次分配 256, 257, 258, ... 这些码位
            n += 1
    cs = [chr(n) for n in cs]  # 把整数码位转换成实际的 Unicode 字符
    return dict(zip(bs, cs))  # 返回 {字节值: 字符} 的映射字典


def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).

    【中文说明】
    作用:给定一个"符号元组"(word，例如把单词拆成单字符的元组)，
    返回其中所有"相邻符号对"组成的集合。这是 BPE 算法每一轮迭代的基础操作——
    BPE 每次都要找出当前"出现频率最高/优先级最高"的相邻符号对进行合并。
    参数:
        word: tuple，由若干"符号"(可以是单字符，也可以是之前已合并出的子词片段)组成，
              代表当前正在处理的一个 token 的分解形式。
    返回值:
        set，元素为 (前一个符号, 后一个符号) 这样的二元组，表示 word 中所有相邻的符号对。
        注意是 set(集合)而不是 list，因为同一个 pair 可能在 word 中重复出现多次，
        BPE 只关心"是否存在"这种 pair，不关心具体出现几次。
    """
    pairs = set()
    prev_char = word[0]  # 从第一个符号开始，逐步与后面的符号两两配对
    for char in word[1:]:
        pairs.add((prev_char, char))  # 把 (前一个符号, 当前符号) 这一相邻对加入集合
        prev_char = char  # 滑动窗口:当前符号变成下一轮的"前一个符号"
    return pairs


class Encoder:
    """
    【中文说明】
    GPT-2 官方 BPE 分词器的核心类。

    整体职责:
        - encode(text)：把一段原始文本字符串编码成一串整数 token id 列表；
        - decode(tokens)：把一串整数 token id 列表还原（解码）回原始文本字符串；
        - bpe(token)：对单个"预分词"片段执行字节对编码合并算法，得到用空格分隔的子词序列。

    内部依赖的核心数据结构:
        - self.encoder / self.decoder：子词字符串 <-> token id 的正反向词表；
        - self.byte_encoder / self.byte_decoder：原始字节 <-> 可打印 Unicode 字符 的正反向映射
          （见 bytes_to_unicode()）；
        - self.bpe_ranks：合并规则的优先级表，记录每一对 (symbol1, symbol2) 在训练时
          被合并的先后顺序（数值越小表示优先级越高，即在训练语料中越常见/越早被合并）；
        - self.cache：对已经处理过的 token 做结果缓存，避免重复执行 BPE 合并循环。
    """
    def __init__(self, encoder, bpe_merges, errors="replace"):
        """
        初始化 Encoder。

        参数:
            encoder: dict，形如 {子词字符串: token_id}，即 encoder.json 加载后的内容，
                     是完整的 BPE 词表（子词 -> id 的映射）。
            bpe_merges: list[tuple[str, str]]，按训练时的合并顺序排列的"合并规则"列表，
                        每一项是一对被合并的符号 (first, second)，列表顺序即优先级顺序
                        （越靠前的合并规则优先级越高，越先被应用）。
            errors: str，解码 UTF-8 字节流失败时的处理策略，传给 bytes.decode(errors=...)，
                    默认 "replace" 表示用替换字符代替无法解码的字节，不抛异常。
        """
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}  # 反转词表，得到 id -> 子词字符串 的映射，供 decode 使用
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()  # 构建 字节 -> 可打印字符 的映射表(见上面函数说明)
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}  # 反向映射：可打印字符 -> 字节，供 decode 还原原始字节用
        # 把 bpe_merges 列表转换成 {合并规则: 优先级序号} 的字典，
        # 序号越小代表这条合并规则在训练时出现得越早/越常用，即优先级越高
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}  # 缓存 token -> BPE 处理结果，避免对相同 token 重复执行合并循环，提升编码速度

        # Should have added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        # 【中文说明】这是"预分词"（pre-tokenization）用的正则表达式，作用是把原始文本
        # 先粗略切分成若干片段，再对每个片段分别执行 BPE 合并。规则依次是：
        #   's|'t|'re|'ve|'m|'ll|'d  —— 常见英文缩写的词尾（如 "'s", "'re"）单独成一段；
        #   ' ?\p{L}+                —— 可选的前导空格 + 一段连续的字母(Letter)；
        #   ' ?\p{N}+                —— 可选的前导空格 + 一段连续的数字(Number)；
        #   ' ?[^\s\p{L}\p{N}]+      —— 可选的前导空格 + 一段连续的"非空白、非字母、非数字"符号(标点等)；
        #   \s+(?!\S)                —— 一段空白，且后面不再紧跟非空白字符（用于吞掉结尾多余空白）；
        #   \s+                      —— 兜底：任意一段空白。
        # 这样可以把"单词/数字/标点/空白"分开处理，避免 BPE 合并跨越这些边界。
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token):
        """
        对单个"预分词片段"(token，一个已经按 self.pat 切分出来、并经过字节->可打印字符
        映射后的字符串)执行字节对编码合并算法。

        算法思路(经典 BPE 合并循环):
            1. 把 token 拆成单字符组成的元组 word；
            2. 反复找出 word 中"优先级最高"(即 self.bpe_ranks 中排名最靠前，
               对应训练时最常见)的相邻符号对 bigram，并把它们合并成一个新符号；
            3. 直到 word 中不再存在任何一条已知的合并规则，或者 word 只剩一个符号为止；
            4. 最终把合并结果用空格拼接成字符串返回(空格是子词之间的分隔符，
               方便后续用 split(" ") 还原成子词列表)。

        参数:
            token: str，单个预分词片段（内部字符已经是"可打印 Unicode 字符"形式，
                   而非原始字节）。
        返回值:
            str，各子词片段以空格分隔拼接后的字符串，例如 "un used"。
        """
        if token in self.cache:
            return self.cache[token]  # 命中缓存，直接复用之前的合并结果，避免重复计算
        word = tuple(token)  # 先把字符串拆成"单字符元组"，作为 BPE 合并的初始状态
        pairs = get_pairs(word)  # 取出所有相邻符号对

        if not pairs:
            return token  # 长度为 0 或 1 的 token 没有相邻对，无需合并，直接原样返回

        while True:
            # 在当前所有相邻符号对中，找出"优先级最高"的一个——
            # 即在 bpe_ranks 中数值最小(训练时最早/最常被合并)的 pair；
            # 若某个 pair 不在 bpe_ranks 中，则赋予 float("inf")，视为最低优先级
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break  # 说明剩余符号对都不存在合并规则，合并过程结束
            first, second = bigram
            new_word = []
            i = 0
            # 下面这段循环:遍历 word，把所有出现的 (first, second) 相邻对合并成 first+second，
            # 其余符号原样保留，构造出合并一轮后的新 word
            while i < len(word):
                try:
                    j = word.index(first, i)  # 从位置 i 开始查找下一个 first 出现的位置
                    new_word.extend(word[i:j])  # 把 i 到 j 之间(不含 j)的符号原样加入结果
                    i = j
                except ValueError:
                    new_word.extend(word[i:])  # 后面已经没有 first 了，剩余部分原样加入
                    break

                # 检查当前位置是否恰好是 (first, second) 这一对，若是则合并为一个新符号
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)  # 合并:两个符号拼接成一个更长的子词符号
                    i += 2  # 跳过这两个已合并的符号
                else:
                    new_word.append(word[i])  # 不满足合并条件，单个符号原样保留
                    i += 1
            new_word = tuple(new_word)
            word = new_word  # 用合并后的新 word 替换旧 word，进入下一轮迭代
            if len(word) == 1:
                break  # 已经合并成单个符号，无法再有相邻对，结束循环
            else:
                pairs = get_pairs(word)  # 基于新 word 重新计算相邻符号对，为下一轮合并做准备
        word = " ".join(word)  # 把最终的符号元组用空格拼接成字符串，作为子词序列的文本表示
        self.cache[token] = word  # 缓存结果，下次遇到相同 token 直接复用
        return word

    def encode(self, text):
        """
        把一段原始文本编码为 token id 列表。

        完整流程:
            原始文本
              -> (1) 用 self.pat 正则做预分词，切成若干片段(单词/标点/空白等)
              -> (2) 每个片段先编码为 UTF-8 字节，再通过 byte_encoder 映射成
                     "可打印 Unicode 字符" 组成的字符串
              -> (3) 对该字符串执行 self.bpe(...) 做 BPE 合并，得到以空格分隔的子词序列
              -> (4) 通过 self.encoder 词表把每个子词字符串映射为整数 token id
              -> (5) 所有片段产生的 id 依次拼接，得到最终的 token id 列表

        参数:
            text: str，原始输入文本。
        返回值:
            list[int]，编码后的 token id 序列，长度等于文本切分出的子词总数
            （通常等价于送入 Transformer 模型的输入序列长度，即形状 (seq_len,)，
            后续再由框架加上 batch 维度变成 (batch, seq_len)）。
        """
        bpe_tokens = []
        for token in re.findall(self.pat, text):  # 用预分词正则把文本切成若干原始片段
            # 先把片段编码为 UTF-8 字节序列，再逐字节查表映射成可打印 Unicode 字符；
            # 这样即使原始字节里含有空格、换行等不可见字符，也能安全地参与后续的字符串处理与 BPE 合并
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            # 对映射后的字符串执行 BPE 合并，得到空格分隔的子词序列，再逐个查词表转成 id
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def decode(self, tokens):
        """
        把 token id 列表解码还原为原始文本字符串，是 encode 的逆过程。

        参数:
            tokens: list[int] 或可迭代的整数序列，模型输出/输入的 token id 序列。
        返回值:
            str，还原出的文本字符串。
        """
        # 第一步:把每个 token id 通过 self.decoder 转回子词字符串(仍是"可打印 Unicode 字符"形式)，
        # 然后拼接成一个长字符串
        text = "".join([self.decoder[token] for token in tokens])
        # 第二步:把每个"可打印 Unicode 字符"通过 byte_decoder 还原为原始字节值，
        # 组装成 bytearray，再按 UTF-8 解码回真正的文本；
        # errors=self.errors 控制遇到无法正确解码的字节时的处理方式(默认用替换字符代替，不报错)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text


def get_encoder(model_name, models_dir):
    """
    从本地磁盘加载指定模型的 BPE 词表文件，构建并返回一个 Encoder 实例。

    参数:
        model_name: str，模型名称，用作子目录名（例如 "117M"），
                    对应的词表文件应位于 models_dir/model_name/ 目录下。
        models_dir: str，存放各个模型词表文件的根目录路径。
    返回值:
        Encoder 实例，已加载好 encoder.json 中的词表以及 vocab.bpe 中的合并规则，
        可直接调用 .encode(text) / .decode(tokens) 使用。
    """
    # encoder.json：JSON 格式，内容是 {子词字符串: token_id} 的完整词表
    with open(os.path.join(models_dir, model_name, "encoder.json"), "r") as f:
        encoder = json.load(f)
    # vocab.bpe：纯文本文件，每一行是一条"合并规则"，形如 "t h"，表示把符号 t 和 h 合并；
    # 文件按训练时的合并顺序排列，顺序本身就代表了优先级
    with open(os.path.join(models_dir, model_name, "vocab.bpe"), "r", encoding="utf-8") as f:
        bpe_data = f.read()
    # 按行切分后，去掉首行(通常是版本说明注释)和末尾的空行，
    # 再把每一行按空白切分成 (first, second) 元组，得到有序的合并规则列表
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split("\n")[1:-1]]
    return Encoder(encoder=encoder, bpe_merges=bpe_merges)


def download_vocab():
    """
    从 OpenAI 官方公开存储桶下载 GPT-2 (117M 版本) 的词表文件
    （encoder.json 和 vocab.bpe），保存到当前目录下的 "gpt2_model" 子目录中，
    供 get_encoder() 后续加载使用。下载过程中会用 tqdm 显示进度条。

    参数: 无
    返回值: 无（副作用是在本地磁盘写入两个词表文件）
    """
    # Modified code from
    subdir = "gpt2_model"
    if not os.path.exists(subdir):
        os.makedirs(subdir)  # 目标目录不存在则创建
    subdir = subdir.replace("\\", "/")  # needed for Windows  # 统一路径分隔符，兼容 Windows 系统

    for filename in ["encoder.json", "vocab.bpe"]:
        # stream=True：流式下载，不一次性把整个文件内容读入内存，适合较大文件
        r = requests.get("https://openaipublic.blob.core.windows.net/gpt-2/models/117M/" + filename, stream=True)

        with open(os.path.join(subdir, filename), "wb") as f:
            file_size = int(r.headers["content-length"])  # 从响应头获取文件总大小，用于进度条显示总量
            chunk_size = 1000
            with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
                # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                # 【中文说明】按 1000 字节为一块，边下载边写入磁盘，并同步更新进度条，
                # 避免一次性占用过多内存
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)




