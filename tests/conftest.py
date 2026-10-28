"""Stub the heavy/optional libraries so the tests run with no GPU, DB or Ollama."""
import os, sys, types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("SESSION_SECRET", "test-secret")
os.environ.setdefault("DATABASE_URL", "postgresql://u:p@localhost/x")
os.environ.setdefault("APP_API_KEY", "test-api-key")

st = types.ModuleType("sentence_transformers")
class _Model:
    def __init__(self, *a, **k): pass
    def get_embedding_dimension(self): return 384
    def encode(self, texts, **k):
        import numpy as np
        return np.zeros((len(texts), 384))
st.SentenceTransformer = _Model
sys.modules.setdefault("sentence_transformers", st)
