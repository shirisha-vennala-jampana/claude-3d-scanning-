"""scan3d: build a 3D world from video/images (or RGB-D captures) and embed metadata in it."""
from .scene import Annotation, Camera, Scene

__all__ = ["Annotation", "Camera", "Scene"]
__version__ = "0.1.0"
