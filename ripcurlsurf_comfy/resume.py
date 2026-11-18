"""Resume nodes - a film of any length is rendered in as many runs as it takes.

Every picture (reference, baked shot, first frame) and every clip of the film is saved by one of these nodes instead
of a plain Save node. Before anything is made, the node looks in ComfyUI/output for the file it would write:

  * the file is there and was made with the same settings  -> it is loaded, and nothing upstream of this node runs
    (ComfyUI's lazy inputs: the image model / video model for that picture or clip are never executed);
  * the file is missing, or the settings changed           -> the upstream nodes run, the result is saved, and a
    small sidecar file (<name>.resume.txt) records the settings fingerprint next to it.

So the same workflow can be queued again after a crash, a power cut or a stop, and it carries on where it stopped;
a chapter workflow of a long film finds the references, bakes and gold frame that an earlier chapter made; and a
scene whose prompt or seed was changed (and every scene that continues from it: the fingerprint of a chained scene
includes the fingerprint of the scene it starts from) is rendered again, while everything else is kept.

  Resume Picture   output/<prefix>_00001_.png   (same file name as the Save Image nodes)   image/reference
  Resume Clip      output/<prefix>_00001_.mp4   (same file name as the Save Video node)    video/ripcurlsurf

mode: "skip if saved" (normal) or "always make" (render again and replace the saved file; the old file is renamed
to <name>.old-<time>). Both write one line to ComfyUI/output/movie_log.txt saying whether the file was reused.
"""
import os
import re
import time
import hashlib

import numpy as np

try:                                   # the one log file (output/movie_log.txt)
    from .qualitylock import movie_log as _movie_log
except Exception:                      # pragma: no cover
    _movie_log = None

MODES = ["skip if saved", "always make"]
SIDE = ".resume.txt"


def _out_dir():
    try:
        import folder_paths
        return folder_paths.get_output_directory()
    except Exception:
        return os.path.abspath("output")


def _log(node, label, verdict, text):
    if _movie_log is None:
        print("[ripcurlsurf_comfy] %s %s: %s - %s" % (node, label, verdict, text))
        return
    try:
        _movie_log(node, label, verdict, text)
    except Exception:
        pass


def _clean_prefix(prefix):
    p = str(prefix or "").strip().replace("\\", "/")
    p = re.sub(r"^/+", "", p)
    if not p or ".." in p.split("/"):
        return None
    return p


def _fingerprint(text):
    return hashlib.sha1(str(text or "").encode("utf-8", "replace")).hexdigest()[:16]


def _find_saved(prefix, ext):
    """The saved file for this prefix: <prefix>_00001_.<ext> (the lowest counter), else <prefix>.<ext>."""
    p = _clean_prefix(prefix)
    if p is None:
        return None, None
    base = os.path.join(_out_dir(), p)
    d, stem = os.path.dirname(base), os.path.basename(base)
    if not os.path.isdir(d):
        return None, base
    rx = re.compile(r"^" + re.escape(stem) + r"_(\d{5})_\." + re.escape(ext) + r"$")
    best = None
    try:
        for n in os.listdir(d):
            m = rx.match(n)
            if m and (best is None or int(m.group(1)) < best[0]):
                best = (int(m.group(1)), os.path.join(d, n))
    except Exception:
        pass
    if best:
        return best[1], base
    plain = base + "." + ext
    return (plain if os.path.isfile(plain) else None), base


def _side_fp(path):
    try:
        with open(path + SIDE, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("fingerprint="):
                    return line.split("=", 1)[1].strip()
    except Exception:
        return None
    return None


def _write_side(path, fp, label):
    try:
        with open(path + SIDE, "w", encoding="utf-8") as f:
            f.write("fingerprint=%s\nlabel=%s\nsaved=%s\n" % (fp, label, time.strftime("%Y-%m-%d %H:%M:%S")))
    except Exception:
        pass


def _reusable(prefix, ext, fingerprint, mode):
    """(path or None, reason). A saved file without a sidecar (made by a plain Save node) is reused as it is."""
    if mode != MODES[0]:
        return None, "always make"
    path, _ = _find_saved(prefix, ext)
    if not path:
        return None, "not saved yet"
    have = _side_fp(path)
    want = _fingerprint(fingerprint)
    if have is None or not str(fingerprint or "").strip() or have == want:
        return path, "saved earlier"
    return None, "settings changed since it was saved"


def _new_path(prefix, ext):
    """<prefix>_00001_.<ext>, like ComfyUI's Save nodes; an existing file is kept as .old-<time>."""
    p = _clean_prefix(prefix)
    base = os.path.join(_out_dir(), p)
    os.makedirs(os.path.dirname(base), exist_ok=True)
    path = base + "_00001_." + ext
    if os.path.exists(path):
        try:
            old = path + ".old-" + time.strftime("%Y%m%d-%H%M%S")
            os.replace(path, old)
            if os.path.exists(path + SIDE):
                os.replace(path + SIDE, old + SIDE)
        except Exception:
            pass
    return path


class ResumePicture:
    """Save a picture under output/<prefix>, or load it when it was saved earlier with the same settings."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prefix": ("STRING", {"default": "ref/reference_1", "multiline": False,
                                      "tooltip": "Path under ComfyUI/output without the extension (ref/maya, "
                                                 "first_frames/film_first_03). Saved as <prefix>_00001_.png."}),
                "fingerprint": ("STRING", {"default": "", "multiline": False,
                                           "tooltip": "The settings this picture is made from (prompt, seed, size). "
                                                      "When they differ from the saved file's, it is made again."}),
                "mode": (MODES, {"tooltip": "skip if saved: reuse the saved file. always make: make it again and "
                                            "replace the file (the old one is renamed .old-<time>)."}),
                "label": ("STRING", {"default": "", "multiline": False, "tooltip": "Name in the log."}),
            },
            "optional": {
                "image": ("IMAGE", {"lazy": True, "tooltip": "The new picture; only made when it is needed."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")
    FUNCTION = "run"
    CATEGORY = "image/reference"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, prefix, fingerprint, mode, label, image=None):
        path, why = _reusable(prefix, "png", fingerprint, mode)
        if path:
            try:
                return "%s:%s" % (path, os.path.getmtime(path))
            except Exception:
                return path
        return float("nan")

    def check_lazy_status(self, prefix, fingerprint, mode, label, image=None):
        path, _ = _reusable(prefix, "png", fingerprint, mode)
        if path:
            return []
        return ["image"] if image is None else []

    def run(self, prefix, fingerprint, mode, label, image=None):
        from PIL import Image
        lab = label or prefix
        path, why = _reusable(prefix, "png", fingerprint, mode)
        if path:
            with Image.open(path) as im:
                a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
            try:
                import torch
                out = torch.from_numpy(a)[None]
            except Exception:
                out = a[None]
            rep = "reused %s (%s)" % (os.path.relpath(path, _out_dir()), why)
            _log("ResumePicture", lab, "REUSED", rep)
            return {"ui": {"text": [rep]}, "result": (out, rep)}
        if image is None:
            raise RuntimeError("Resume Picture '%s': nothing is saved under output/%s and no picture is connected." % (lab, prefix))
        if _clean_prefix(prefix) is None:
            raise RuntimeError("Resume Picture '%s': bad prefix %r" % (lab, prefix))
        img = image
        try:
            arr = img[0].detach().cpu().numpy() if hasattr(img, "detach") else np.asarray(img)[0]
        except Exception:
            arr = np.asarray(img)[0]
        path = _new_path(prefix, "png")
        tmp = path + ".tmp.png"
        Image.fromarray(np.clip(np.asarray(arr, dtype=np.float32)[..., :3] * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(tmp)
        os.replace(tmp, path)
        _write_side(path, _fingerprint(fingerprint), lab)
        rep = "made and saved %s (%s)" % (os.path.relpath(path, _out_dir()), why)
        _log("ResumePicture", lab, "SAVED", rep)
        return {"ui": {"text": [rep]}, "result": (image, rep)}


def _video_from_file(path):
    try:
        from comfy_api.latest import InputImpl
        return InputImpl.VideoFromFile(path)
    except Exception:
        pass
    try:
        from comfy_api.input_impl import VideoFromFile
        return VideoFromFile(path)
    except Exception as e:
        raise RuntimeError("This ComfyUI has no VideoFromFile (update ComfyUI): %s" % e)


class ResumeClip:
    """Save a clip under output/<prefix>_00001_.mp4, or load it when it was saved earlier with the same settings."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prefix": ("STRING", {"default": "video/film_scene_01", "multiline": False,
                                      "tooltip": "Path under ComfyUI/output without the extension. "
                                                 "Saved as <prefix>_00001_.mp4, like Save Video."}),
                "fingerprint": ("STRING", {"default": "", "multiline": False,
                                           "tooltip": "The settings this clip is made from (prompt, seed, size, steps "
                                                      "and the fingerprint of the clip it continues from). When they "
                                                      "differ from the saved file's, the clip is rendered again."}),
                "mode": (MODES, {"tooltip": "skip if saved: reuse the saved clip. always make: render again and "
                                            "replace the file (the old one is renamed .old-<time>)."}),
                "label": ("STRING", {"default": "", "multiline": False, "tooltip": "Name in the log."}),
            },
            "optional": {
                "video": ("VIDEO", {"lazy": True, "tooltip": "The new clip; only rendered when it is needed."}),
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "report")
    FUNCTION = "run"
    CATEGORY = "video/ripcurlsurf"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, prefix, fingerprint, mode, label, video=None):
        path, why = _reusable(prefix, "mp4", fingerprint, mode)
        if path:
            try:
                return "%s:%s" % (path, os.path.getmtime(path))
            except Exception:
                return path
        return float("nan")

    def check_lazy_status(self, prefix, fingerprint, mode, label, video=None):
        path, _ = _reusable(prefix, "mp4", fingerprint, mode)
        if path:
            return []
        return ["video"] if video is None else []

    def run(self, prefix, fingerprint, mode, label, video=None):
        lab = label or prefix
        path, why = _reusable(prefix, "mp4", fingerprint, mode)
        if path:
            rep = "reused %s (%s)" % (os.path.relpath(path, _out_dir()), why)
            _log("ResumeClip", lab, "REUSED", rep)
            return {"ui": {"text": [rep]}, "result": (_video_from_file(path), rep)}
        if video is None:
            raise RuntimeError("Resume Clip '%s': nothing is saved under output/%s and no video is connected." % (lab, prefix))
        if _clean_prefix(prefix) is None:
            raise RuntimeError("Resume Clip '%s': bad prefix %r" % (lab, prefix))
        path = _new_path(prefix, "mp4")
        tmp = path + ".tmp.mp4"
        try:
            video.save_to(tmp)
        except TypeError:
            video.save_to(tmp, None, None)
        os.replace(tmp, path)
        _write_side(path, _fingerprint(fingerprint), lab)
        rep = "rendered and saved %s (%s)" % (os.path.relpath(path, _out_dir()), why)
        _log("ResumeClip", lab, "SAVED", rep)
        # the clip the rest of the graph uses is the saved file, so a re-queued run and this run see the same pixels
        return {"ui": {"text": [rep]}, "result": (_video_from_file(path), rep)}


NODE_CLASS_MAPPINGS = {"ResumePicture": ResumePicture, "ResumeClip": ResumeClip}
NODE_DISPLAY_NAME_MAPPINGS = {"ResumePicture": "Resume Picture (reuse if saved)", "ResumeClip": "Resume Clip (reuse if saved)"}
