"""ripcurlsurf_comfy - ComfyUI custom nodes for the long-video generator.

Install: the folder must end up exactly as
    ComfyUI/custom_nodes/ripcurlsurf_comfy/__init__.py
    ComfyUI/custom_nodes/ripcurlsurf_comfy/refimagetools.py
    ComfyUI/custom_nodes/ripcurlsurf_comfy/sceneguard.py
(not a folder inside a folder), then restart ComfyUI. Look in the ComfyUI console for the
line  [ripcurlsurf_comfy] loaded ...  - if a module fails, the reason is printed there.
Nodes:
  Reference Photo Finish (remove painted look)            image/reference
  Load Image (optional, falls back to original)           image/reference
  Scene Guard (cut off mid-clip scene changes)            video/ripcurlsurf
"""
import importlib
import traceback

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for _mod in ("refimagetools", "sceneguard"):
    try:
        _m = importlib.import_module("." + _mod, __name__)
        NODE_CLASS_MAPPINGS.update(_m.NODE_CLASS_MAPPINGS)
        NODE_DISPLAY_NAME_MAPPINGS.update(_m.NODE_DISPLAY_NAME_MAPPINGS)
    except Exception:
        print("[ripcurlsurf_comfy] FAILED to load %s.py - the other module still loads. Reason:" % _mod)
        traceback.print_exc()

print("[ripcurlsurf_comfy] loaded nodes: " + (", ".join(sorted(NODE_CLASS_MAPPINGS)) or "NONE"))

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
