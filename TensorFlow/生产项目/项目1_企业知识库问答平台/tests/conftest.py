import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("KB_EMBED_BACKEND", "stub")
os.environ.setdefault("KB_LLM_BACKEND", "stub")
