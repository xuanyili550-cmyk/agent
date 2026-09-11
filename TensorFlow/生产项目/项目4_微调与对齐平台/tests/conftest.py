import os, sys, tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("FT_REGISTRY_DIR", tempfile.mkdtemp(prefix="ft_test_"))
