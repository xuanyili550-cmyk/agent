"""QC 报告子包：导出 QCReport 及其组成模型（QCDecision / QCItemScore / QCThresholds），
以及两个入口函数——build_qc_report（纯聚合）和 run_full_qc（端到端跑四项检查再聚合）。"""

from .qc_report import (
    QCDecision,
    QCItemScore,
    QCReport,
    QCThresholds,
    build_qc_report,
    run_full_qc,
)

__all__ = [
    "QCDecision",
    "QCItemScore",
    "QCReport",
    "QCThresholds",
    "build_qc_report",
    "run_full_qc",
]
