"""Importing/installing this package does not enable the backend."""
import importlib.metadata
import os

__version__ = "0.1.0"


def register():
    if os.environ.get("INQUANT_VLLM") != "1":
        return None
    # Platform discovery catches exceptions from plugin callbacks. Return the
    # class path here and check compatibility when that class is resolved, so
    # an incompatible version cannot silently fall back to ordinary CUDA.
    return "inquant_vllm.platform.InQuantPlatform"


def check_version():
    version = importlib.metadata.version("vllm")
    if version != "0.9.1":
        raise RuntimeError(f"inquant-vllm requires vllm==0.9.1, found {version}")
