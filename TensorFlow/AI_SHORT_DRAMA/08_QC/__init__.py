"""08_QC 质检层顶层包：不导出任何符号，只让 character / scene / video / audio / reports 子包能以相对导入互相引用。

流水线位置：07_GENERATION 生成素材之后、09_POST 后期之前的"Generate -> QC -> PASS? -> Retry/Approved" 闸门。
故意留空是为了保持顶层 import 轻量——各子包按需再加载 torch / opencv / pydub 等重依赖。
"""
