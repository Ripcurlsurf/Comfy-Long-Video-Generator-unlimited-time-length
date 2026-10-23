"""ripcurlsurf_comfy - ComfyUI custom nodes for the long-video generator.

Install: the folder must end up exactly as
    ComfyUI/custom_nodes/ripcurlsurf_comfy/__init__.py
    ComfyUI/custom_nodes/ripcurlsurf_comfy/refimagetools.py
    ComfyUI/custom_nodes/ripcurlsurf_comfy/sceneguard.py
    ComfyUI/custom_nodes/ripcurlsurf_comfy/qualitylock.py
    ComfyUI/custom_nodes/ripcurlsurf_comfy/storykeys.py
    ComfyUI/custom_nodes/ripcurlsurf_comfy/troubleshoot.py
(not a folder inside a folder), then restart ComfyUI. Look in the ComfyUI console for the
line  [ripcurlsurf_comfy] loaded ...  - if a module fails, the reason is printed there.
Nodes:
  Reference Photo Finish (remove painted look)            image/reference
  Load Image (optional, falls back to original)           image/reference
  Scene Anchor (identity-locked continuity)               image/reference
  Scene Guard (exact length, cuts, best of N tries)       video/ripcurlsurf
  Quality Anchor (gold frame for the whole film)          video/ripcurlsurf
  Scene Quality Lock (whole clip, every frame)            video/ripcurlsurf
  Quality Lock (one frame)                                video/ripcurlsurf
  Best Hand-off Frame (sharpest of the last frames)       video/ripcurlsurf
  Quality Gate (the frame the next scene starts from)     video/ripcurlsurf
  Story Keyframe (first or last frame of a scene, saved)  video/ripcurlsurf
  Troubleshooting Log (one clip, for finding problems)    video/ripcurlsurf
Every node writes to one log file, ComfyUI/output/movie_log.txt, in the order things happen.
Version 2026.10.20.1 - the console line below shows the version and every node that loaded.
"""
import importlib
import traceback

VERSION = "2026.10.20.1"

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for _mod in ("refimagetools", "sceneguard", "qualitylock", "storykeys", "troubleshoot"):
    try:
        _m = importlib.import_module("." + _mod, __name__)
        NODE_CLASS_MAPPINGS.update(_m.NODE_CLASS_MAPPINGS)
        NODE_DISPLAY_NAME_MAPPINGS.update(_m.NODE_DISPLAY_NAME_MAPPINGS)
    except Exception:
        print("[ripcurlsurf_comfy] FAILED to load %s.py - the other modules still load. Reason:" % _mod)
        traceback.print_exc()

print("[ripcurlsurf_comfy] version %s, loaded nodes: %s" % (VERSION, ", ".join(sorted(NODE_CLASS_MAPPINGS)) or "NONE"))

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
