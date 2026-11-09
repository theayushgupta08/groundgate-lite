"""GroundGate Lite: open-source hybrid RAG with an "I don't know" gate."""

from .config import Settings, get_settings
from .engine import Lite
from .types import Document

__version__ = "1.0.0"
__all__ = ["Lite", "Settings", "get_settings", "Document"]
