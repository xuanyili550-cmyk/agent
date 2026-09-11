import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# 测试用独立索引目录，避免污染真实数据(必须在导入 app 前设好，因 get_settings 会缓存)
os.environ.setdefault("EMB_INDEX_DIR", tempfile.mkdtemp(prefix="emb_test_"))
