"""ripcurlsurf_comfy - ComfyUI custom nodes for the long-video generator.

Install: put this whole folder in ComfyUI/custom_nodes/ and restart ComfyUI.
Nodes (category image/reference):
  Reference Photo Finish (remove painted look)
  Load Image (optional, falls back to original)
"""
from .refimagetools import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
