"""结构化 JSON 行日志(生产便于采集)。"""
import json
import logging
import sys
import time


class JsonFormatter(logging.Formatter):
    def format(self, record):
        p = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
             "level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        if record.exc_info:
            p["exc"] = self.formatException(record.exc_info)
        return json.dumps(p, ensure_ascii=False)


def setup_logging(debug=False):
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [h]
    root.setLevel(logging.DEBUG if debug else logging.INFO)


def get_logger(name):
    return logging.getLogger(name)
