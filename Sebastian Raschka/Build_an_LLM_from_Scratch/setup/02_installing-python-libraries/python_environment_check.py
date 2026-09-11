# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""环境自检脚本（配套《从零构建大语言模型》一书的 setup 部分）。

本脚本用于在开始学习前检查本地 Python 环境是否满足要求：
1. Python 版本是否 >= 3.9；
2. requirements.txt 中列出的各依赖包是否已安装、且版本满足指定的版本约束。
运行方式：`python python_environment_check.py`，输出每一项的 [OK] / [FAIL] 结果。
"""

from importlib.metadata import PackageNotFoundError, import_module, version as get_version
from os.path import dirname, exists, join, realpath
from packaging.version import parse as version_parse
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
import platform
import sys

# 检查当前 Python 版本：低于 3.9 给出 [FAIL] 提示，否则打印 [OK]
if version_parse(platform.python_version()) < version_parse("3.9"):
    print("[FAIL] We recommend Python 3.9 or newer but found version %s" % sys.version)
else:
    print("[OK] Your Python version is %s" % platform.python_version())


def get_packages(pkgs):
    """
    Returns a dictionary mapping package names (in lowercase) to their installed version.

    中文说明：返回一个字典，键为包名（小写），值为已安装的版本号。
    对每个包依次尝试多种导入方式（模块名映射、连字符转下划线等），
    先读模块的 __version__，读不到再用 importlib.metadata.version 获取；
    若最终仍找不到，则记为 "0.0"（表示未安装/未知版本）。
    """
    # 某些发行包名与实际可导入的模块名不一致，这里做手动映射（如 tensorflow-cpu 实际模块是 tensorflow）
    PACKAGE_MODULE_OVERRIDES = {
        "tensorflow-cpu": ["tensorflow", "tensorflow_cpu"],
    }
    result = {}
    for p in pkgs:
        # Determine possible module names to try.
        # 确定该包可能对应的模块名列表（优先用上面的手动映射，否则就用包名本身）
        module_names = PACKAGE_MODULE_OVERRIDES.get(p.lower(), [p])
        version_found = None
        for module_name in module_names:
            try:
                # 先尝试直接 import 该模块，并读取其 __version__ 属性
                imported = import_module(module_name)
                version_found = getattr(imported, "__version__", None)
                if version_found is None:
                    # 模块没有 __version__ 属性时，退而用包元数据查询版本
                    try:
                        version_found = get_version(module_name)
                    except PackageNotFoundError:
                        version_found = None
                if version_found is not None:
                    break  # Stop if we successfully got a version.（成功拿到版本就不再尝试其它名字）
            except ImportError:
                # Also try replacing hyphens with underscores as a fallback.
                # 导入失败时的兜底：把连字符换成下划线再试一次（如 some-pkg -> some_pkg）
                alt_module = module_name.replace("-", "_")
                if alt_module != module_name:
                    try:
                        imported = import_module(alt_module)
                        version_found = getattr(imported, "__version__", None)
                        if version_found is None:
                            try:
                                version_found = get_version(alt_module)
                            except PackageNotFoundError:
                                version_found = None
                        if version_found is not None:
                            break
                    except ImportError:
                        continue
                continue
        if version_found is None:
            version_found = "0.0"  # 所有尝试都失败：视为未安装/未知，用 "0.0" 占位
        result[p.lower()] = version_found
    return result


def get_requirements_dict():
    """
    Parses requirements.txt and returns a dictionary mapping package names (in lowercase)
    to specifier strings (e.g. ">=2.18.0,<3.0"). It uses the Requirement class from
    packaging.requirements to properly handle environment markers, and converts each object's
    specifier to a string.

    中文说明：解析 requirements.txt，返回「包名(小写) -> 版本约束字符串」的字典
    （如 ">=2.18.0,<3.0"）。用 packaging 的 Requirement 类正确处理环境标记（marker），
    并把每个需求对象的版本约束转成字符串。
    """

    # 定位 requirements.txt：先找项目根目录（当前文件往上两级），找不到再退回当前目录
    PROJECT_ROOT = dirname(realpath(__file__))
    PROJECT_ROOT_UP_TWO = dirname(dirname(PROJECT_ROOT))
    REQUIREMENTS_FILE = join(PROJECT_ROOT_UP_TWO, "requirements.txt")
    if not exists(REQUIREMENTS_FILE):
        REQUIREMENTS_FILE = join(PROJECT_ROOT, "requirements.txt")

    reqs = {}
    with open(REQUIREMENTS_FILE) as f:
        for line in f:
            # Remove inline comments and trailing whitespace.
            # This splits on the first '#' and takes the part before it.
            # 去掉行内注释和首尾空白：以第一个 '#' 为界，只取其前面的部分
            line = line.split("#", 1)[0].strip()
            if not line:
                continue  # 跳过空行（或整行都是注释的行）
            try:
                req = Requirement(line)  # 解析成 Requirement 对象（自动处理版本约束与环境标记）
            except Exception as e:
                print(f"Skipping line due to parsing error: {line} ({e})")
                continue
            # Evaluate the marker if present.
            # 若存在环境标记（如 ; python_version < "3.10"）且当前环境不满足，则跳过该依赖
            if req.marker is not None and not req.marker.evaluate():
                continue
            # Store the package name and its version specifier.
            # 记录「包名 -> 版本约束」；没有显式约束时用 ">=0"（表示任意版本均可）
            spec = str(req.specifier) if req.specifier else ">=0"
            reqs[req.name.lower()] = spec
    return reqs


def check_packages(reqs):
    """
    Checks the installed versions of packages against the requirements.

    中文说明：把「已安装版本」与「requirements 要求的版本约束」逐一对比，
    满足约束打印 [OK]，不满足打印 [FAIL] 并提示应安装的版本范围。
    """
    installed = get_packages(reqs.keys())  # 一次性查询所有依赖的已安装版本
    for pkg_name, spec_str in reqs.items():
        spec_set = SpecifierSet(spec_str)  # 把约束字符串转成可用于版本比较的 SpecifierSet
        actual_ver = installed.get(pkg_name, "0.0")
        if actual_ver == "N/A":
            continue
        actual_ver_parsed = version_parse(actual_ver)
        # If the installed version is a pre-release, allow pre-releases in the specifier.
        # 若装的是预发布版（如 rc/beta），默认约束会排除它，这里放开以免误报 FAIL
        if actual_ver_parsed.is_prerelease:
            spec_set.prereleases = True
        if actual_ver_parsed not in spec_set:  # 版本不在允许范围内 -> 不满足要求
            print(f"[FAIL] {pkg_name} {actual_ver_parsed}, please install a version matching {spec_set}")
        else:
            print(f"[OK] {pkg_name} {actual_ver_parsed}")


def main():
    """脚本入口：先解析 requirements.txt，再逐项检查已安装依赖的版本是否达标。"""
    reqs = get_requirements_dict()
    check_packages(reqs)


if __name__ == "__main__":
    # 直接运行本文件时执行环境检查（被 import 时不会自动执行）
    main()