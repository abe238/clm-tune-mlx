"""CLM-8B on Apple Silicon: the Qwen3-8B encoder and the CLM heads in MLX, plus head training and a batching server."""
from .embedder import MLXEmbedder, load_engine
from .engine import MLXEngine

__all__ = ["MLXEmbedder", "MLXEngine", "load_engine"]
__version__ = "0.3.0"
