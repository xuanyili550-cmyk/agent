# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# Internal utility functions (not intended for public use)

"""
模块级中文说明：
本文件是《从零构建大语言模型》(Build a Large Language Model From Scratch) 全书配套代码库中的
内部工具模块（仅供内部使用，不作为公共 API 对外暴露）。

主要功能：
1. 从 Jupyter Notebook（.ipynb 文件）中自动提取代码单元里的 import 语句、函数定义和类定义，
   并将它们拼装成一个可执行的 Python 模块（`import_definitions_from_notebook`）。
   这样做的目的是：书中很多代码是分散写在各章节 notebook 里的，测试代码（pytest）需要复用
   这些函数/类，而不必手动把 notebook 转成 .py 文件维护两份代码。
2. 提供一个简单的文件下载工具函数 `download_file`，主要给测试用例下载权重/数据文件使用。

依赖库：
- ast：用于解析 Python 源码的抽象语法树（AST），从而准确提取 import 语句。
- re：用于对提取出的函数源码做正则替换（例如统一 `load_weights_into_xxx` 系列函数的形参名）。
- types：用于动态创建一个新的模块对象（ModuleType），把从 notebook 提取的代码 exec 到该模块的
  命名空间里。
- pathlib.Path：跨平台的路径处理。
- nbformat：读取 .ipynb 文件（Jupyter Notebook 的 JSON 格式）为可编程访问的对象。
- requests：用于 HTTP 下载文件。
"""

import ast
import re
import types
from pathlib import Path

import nbformat
import requests


def _extract_imports(src: str):
    """
    从一段 Python 源代码字符串中提取所有的 import 语句，并以字符串形式返回。

    作用：
        书中 notebook 的每个代码单元里可能都写了各自的 import（例如 `import torch`），
        本函数通过 AST 解析，把这些 import 语句“文本化”地收集出来，方便后续统一拼接到
        动态生成的模块顶部，保证提取出来的函数/类在新模块里能正确引用到这些依赖。

    参数：
        src (str): 单个 notebook 代码单元（cell）的源代码文本。

    返回：
        list[str]: 提取到的 import 语句列表，每一项是形如
            "import torch"
            "import torch.nn as nn"
            "from typing import List, Optional as Opt"
        的字符串。如果源码存在语法错误（无法被 ast.parse 解析），则返回空列表（容错处理，
        避免因为某个 cell 写了不完整/非纯 Python 代码而导致整个提取流程崩溃）。
    """
    out = []
    try:
        # 尝试将源码解析为 AST；notebook 中的某些 cell 可能包含 magic 命令（如 %matplotlib inline）
        # 或不完整代码，会导致 SyntaxError，因此用 try/except 兜底。
        tree = ast.parse(src)
    except SyntaxError:
        return out
    # 只遍历模块顶层语句（tree.body），因为 import 通常写在顶层，不深入函数/类内部查找。
    for node in tree.body:
        if isinstance(node, ast.Import):
            # 形如 `import torch` 或 `import torch as t, os`
            parts = []
            for n in node.names:
                # n.asname 存在时表示使用了 `as` 别名，例如 `import numpy as np`
                parts.append(f"{n.name} as {n.asname}" if n.asname else n.name)
            out.append("import " + ", ".join(parts))
        elif isinstance(node, ast.ImportFrom):
            # 形如 `from torch import nn` 或相对导入 `from . import module`
            module = node.module or ""
            parts = []
            for n in node.names:
                parts.append(f"{n.name} as {n.asname}" if n.asname else n.name)
            # node.level 表示相对导入的层级（点号数量），0 表示绝对导入。
            level = "." * node.level if getattr(node, "level", 0) else ""
            out.append(f"from {level}{module} import " + ", ".join(parts))
    return out


def _extract_defs_and_classes_from_code(src):
    """
    从一段源代码字符串中，只保留函数定义（def / async def）和类定义（class），
    过滤掉其余的顶层语句（例如脚本中的临时变量赋值、打印调用、绘图代码等）。

    作用：
        notebook 的代码单元里往往既有函数/类定义，也有用于演示/调试的临时代码
        （比如实例化模型并打印一下）。本函数通过“基于缩进 + 简单逐行扫描”的方式，
        把 def/class 块完整地摘取出来（包括多行函数签名和装饰器），其余代码丢弃，
        这样拼出来的模块只包含可复用的定义，不会因为顶层的临时代码（如未定义的变量）
        而在 exec 时报错。

    参数：
        src (str): 单个 notebook 代码单元（cell）的源代码文本。

    返回：
        str: 只包含函数/类定义（及其紧邻装饰器）的代码文本；如果该 cell 中函数名匹配
            `load_weights_into_xxx` 这一约定命名模式，还会将其第一个形参统一重命名为
            `model`（详见函数末尾的正则替换逻辑）。
    """
    def _is_header_complete(header_lines):
        """
        判断当前已收集的函数/类“头部”（可能跨多行）是否已经收尊完整。

        参数：
            header_lines (list[str]): 到目前为止收集到的、属于同一个 def/class 头部的行。

        返回：
            bool: True 表示头部已经以冒号结尾且括号已配平（说明签名收集完毕，
                可以开始收集函数体了）；False 表示还需要继续读取下一行。
        """
        header = "\n".join(header_lines).rstrip()
        if not header.endswith(":"):
            # 头部必须以冒号结尾（Python 语法要求），否则一定还没结束。
            return False

        # Track bracket balance for multiline signatures
        # like:
        # def fn(
        #     arg,
        # ):
        # 通过统计三种括号的“开减闭”差值来判断括号是否已配平；
        # 只要还有未闭合的括号（balance > 0），说明签名还没写完，即便当前行以冒号结尾
        # 也可能只是形参默认值里的冒号（例如类型注解 dict 字面量），需要继续读取。
        balance = (
            header.count("(") - header.count(")")
            + header.count("[") - header.count("]")
            + header.count("{") - header.count("}")
        )
        return balance <= 0

    lines = src.splitlines()
    kept = []  # 用于收集最终保留下来的代码行
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        if stripped.startswith("@"):
            # 遇到装饰器行（如 `@torch.no_grad()`），先向下探测紧跟着的非空行
            # 是否是 def/class，只有确认后面确实是函数/类定义时才保留该装饰器，
            # 避免误保留一段不构成函数定义的、以 @ 开头的孤立内容。
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].lstrip().startswith(("def ", "class ", "async def ")):
                kept.append(line)
                i += 1
                continue
        if stripped.startswith(("def ", "class ", "async def ")):
            # 命中函数/类定义的起始行，开始完整摘取该定义块。
            kept.append(line)
            base_indent = len(line) - len(stripped)  # 记录该定义的缩进层级，用于判断函数体边界
            i += 1

            # Handle multiline signatures before consuming the function/class body.
            # 先处理可能跨多行的函数签名（例如参数很多、逐行换行书写的情况），
            # 直到 `_is_header_complete` 判定头部已经完整（括号配平且以冒号收尾）。
            header_lines = [line]
            while i < len(lines) and not _is_header_complete(header_lines):
                header_lines.append(lines[i])
                kept.append(lines[i])
                i += 1

            while i < len(lines):
                nxt = lines[i]
                if nxt.strip() == "":
                    # 空行一律保留（不影响缩进判断，也保持代码可读性）。
                    kept.append(nxt)
                    i += 1
                    continue
                indent = len(nxt) - len(nxt.lstrip())
                if indent <= base_indent and not nxt.lstrip().startswith(("#", "@")):
                    # 缩进回落到函数/类定义同级或更浅，且不是注释/装饰器行，
                    # 说明当前函数/类的函数体已经结束，跳出内层循环，
                    # 外层 while 会在下一轮重新判断这一行是否是新的 def/class。
                    break
                kept.append(nxt)
                i += 1
            continue
        i += 1

    code = "\n".join(kept)

    # General rule:
    # replace functions defined like `def load_weights_into_xxx(ClassName, ...`
    # with `def load_weights_into_xxx(model, ...`
    # 书中不同章节里，“把预训练权重加载进模型”的函数第一个形参命名并不统一
    # （有时用具体类名，有时用别的名字），这里用正则统一改成 `model`，
    # 以便动态生成的模块对外提供一致的调用接口。
    code = re.sub(
        r"(def\s+load_weights_into_\w+\s*\()\s*\w+\s*,",
        r"\1model,",
        code
    )
    return code


def import_definitions_from_notebook(nb_dir_or_path, notebook_name=None, *, extra_globals=None):
    """
    从指定的 Jupyter Notebook 文件中提取所有代码单元的 import 语句以及函数/类定义，
    动态构建并执行成一个新的 Python 模块，返回该模块对象，供测试代码像
    `import` 普通模块一样使用其中定义的函数/类。

    参数：
        nb_dir_or_path (str | Path):
            可以是 notebook 所在的目录路径（此时需配合 `notebook_name` 参数指定文件名），
            也可以直接是 notebook 文件本身的完整路径（此时 `notebook_name` 应为 None）。
        notebook_name (str | None, 可选):
            当 `nb_dir_or_path` 是目录时，指定该目录下具体的 notebook 文件名
            （例如 "ch02.ipynb"）。
        extra_globals (dict | None, 关键字参数):
            额外注入到新建模块全局命名空间中的变量/对象，会在 exec 提取出的代码之前
            先写入模块的 `__dict__`，可用于提供 notebook 代码运行时依赖的外部上下文
            （例如提前放好的配置对象），避免因缺少某些全局变量而导致 exec 失败。

    返回：
        types.ModuleType: 一个新建的、名字来源于 notebook 文件名的模块对象，其命名空间中
            包含了从 notebook 所有代码单元里提取出的 import 依赖以及函数/类定义，可以直接
            通过 `模块对象.函数名` 或 `模块对象.类名` 的方式访问。

    异常：
        FileNotFoundError: 当推导出的 notebook 文件路径不存在时抛出。
    """
    nb_path = Path(nb_dir_or_path)
    if notebook_name is not None:
        # 如果传入的是目录，则拼接目录 + 文件名；否则说明调用方直接把完整路径
        # 塞进了 nb_dir_or_path，此时忽略目录拼接逻辑，直接使用 nb_path。
        nb_file = nb_path / notebook_name if nb_path.is_dir() else nb_path
    else:
        nb_file = nb_path

    if not nb_file.exists():
        raise FileNotFoundError(f"Notebook not found: {nb_file}")

    # 用 nbformat 按版本 4（当前 Jupyter Notebook 的标准格式版本）读取 notebook 文件，
    # 得到一个包含 cells（代码单元/markdown 单元等）的可编程对象。
    nb = nbformat.read(nb_file, as_version=4)

    import_lines = []
    seen = set()  # 用于给 import 语句去重，避免多个 cell 重复 import 同一模块导致冗余
    for cell in nb.cells:
        if cell.cell_type == "code":
            # 只处理代码单元（code cell），忽略 markdown 说明性单元。
            for line in _extract_imports(cell.source):
                if line not in seen:
                    import_lines.append(line)
                    seen.add(line)

    for required in ("import torch", "import torch.nn as nn"):
        # 书中几乎所有章节的函数/类定义都依赖 torch 和 torch.nn，
        # 这里做兜底保证：即便某个 notebook 恰好没有显式写出这两行 import
        # （比如它假设前面的 cell 已经导入过），生成的模块依然能正常引用 torch/nn。
        if required not in seen:
            import_lines.append(required)
            seen.add(required)

    pieces = []
    for cell in nb.cells:
        if cell.cell_type == "code":
            # 对每个代码单元分别提取函数/类定义文本，按 cell 顺序收集，
            # 保持与 notebook 中原始的定义顺序一致。
            pieces.append(_extract_defs_and_classes_from_code(cell.source))

    # 把去重后的 import 语句放在最前面，紧接着依次拼上各个 cell 提取出的定义代码，
    # 用空行分隔，拼装成一份完整、可独立执行的模块源码文本。
    src = "\n\n".join(import_lines + pieces)

    # 模块名基于 notebook 文件名生成：把连字符和空格替换成下划线，使其成为合法的
    # Python 标识符；如果处理后为空字符串（极端情况），退化使用默认名 "notebook_defs"。
    mod_name = nb_file.stem.replace("-", "_").replace(" ", "_") or "notebook_defs"
    mod = types.ModuleType(mod_name)

    if extra_globals:
        # 在 exec 之前先把调用方提供的额外全局变量写入模块命名空间，
        # 这样提取出的函数体在执行时能够引用到这些预置的对象。
        mod.__dict__.update(extra_globals)

    # 关键一步：把拼装好的源码字符串在新模块自己的命名空间（mod.__dict__）中执行，
    # 执行后，源码里定义的所有函数/类都会成为 mod 的属性，效果等价于 `import mod`。
    exec(src, mod.__dict__)
    return mod


def download_file(url, out_dir="."):
    """Simple file download utility for tests.

    中文说明：
        一个简单的文件下载工具函数，主要用于测试代码中按需下载权重文件/数据文件。

    参数：
        url (str): 待下载文件的完整 URL 地址；文件名取自 URL 路径的最后一段
            （通过 `Path(url).name` 得到，不解析查询字符串）。
        out_dir (str, 默认 "."): 下载文件保存的目标目录，若不存在会自动递归创建。

    返回：
        pathlib.Path: 下载完成后（或文件已存在、跳过下载时）本地文件的完整路径。

    异常：
        RuntimeError: 当网络请求出现任何异常（连接失败、超时、HTTP 状态码非 2xx 等）时，
            会捕获原始异常并包装为 RuntimeError 重新抛出，附带原始错误信息，方便上层定位问题。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在，避免写文件时报错
    filename = Path(url).name
    dest = out_dir / filename

    if dest.exists():
        # 文件已存在则直接复用，避免测试反复运行时重复下载、浪费网络和时间。
        return dest

    try:
        # stream=True：以流式方式获取响应体，配合下面的 iter_content 分块写盘，
        # 避免大文件一次性全部读入内存。
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()  # 状态码非 2xx 时主动抛出 HTTPError，交由外层统一处理
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    # chunk 可能为空 bytes（keep-alive 场景下的心跳块），需要过滤掉再写入。
                    f.write(chunk)
        return dest
    except Exception as e:
        # 统一捕获下载过程中的任何异常，转换为更明确的 RuntimeError 并保留原始错误信息，
        # 便于调用方（测试用例）快速定位是哪个 URL 下载失败及具体原因。
        raise RuntimeError(f"Failed to download {url}: {e}")
