"""
storykeys.py - ComfyUI custom node for the Story builder of the long-video generator (package: ripcurlsurf_comfy).

STORY KEYFRAME
One first or last frame of a scene, made by the image model from text (and the reference pictures). The node
  - keeps its look in line with the film: detail, saturation and contrast are measured against the gold frame
    and pulled back if they drifted (no grain is added: keyframes feed the video model),
  - removes a small colour cast or exposure drift compared with the picture it was made from. Chains of image
    edits slowly turn warmer, cooler, brighter or darker; a big, deliberate change (lights going out, sunset)
    is larger than the limits and is left alone,
  - saves the picture to ComfyUI/output/story/<name>.png, replacing the previous version, so a later batch, a
    single-scene redo or the video pass loads exactly this picture,
  - or loads the saved picture instead of making a new one.
  - for a LAST frame: measures how much of the picture changed from the scene's first frame. When the two are almost
    the same, the video model has nothing to animate and the clip looks frozen (or moves, then stops). Then a second
    picture (made with a stronger change, only when needed) is tried and the one with real movement is kept. Very
    big changes (a different camera angle) are flagged too: the video may morph or cut between them.

MODES
  make                       the image model makes the picture; it is checked and saved
  load                       loads output/story/<name>.png; stops with a clear message if the file is missing
  load, or make if missing   loads the saved picture; only when there is none does the image model make it
A PNG in output/story that you replace or retouch by hand is used on the next run (load modes).

The picture inputs are lazy: in a load mode ComfyUI does not run the image model at all, and the second picture of a
last frame is only made when the first one barely differs from the scene's first frame.
Plain numpy, plus Pillow (part of ComfyUI). No model, no extra installs.
"""
import math
import os
import re
import time

import numpy as np
import torch

from .qualitylock import (_Checked, _fmt, _is_blank, _log, _out_dir, _resize, _settings, _to_tensor, apply_correction,
                          plan_correction, profile, quality_flags, quality_score, relative)

try:
    from .refimagetools import _lab_to_rgb, _rgb_to_lab
except Exception:                       # colour lock is skipped if the colour helpers are missing
    _lab_to_rgb = _rgb_to_lab = None

VERSION = "2026.10.16.1"
MODES = ["make", "load", "load, or make if missing"]
_SUB = "story"


# ------------------------------------------------------------------ files
def _paths(name):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name or "")).strip("._") or "keyframe"
    d = os.path.join(_out_dir(), _SUB)
    return d, os.path.join(d, safe + ".png"), safe


def _load_png(path):
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        a = np.asarray(im, dtype=np.float32) / 255.0
    if a.ndim != 3 or a.shape[0] < 16 or a.shape[1] < 16:
        raise ValueError("unexpected picture size %s" % (a.shape,))
    return a


def _save_png(rgb, path):
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.png"
    Image.fromarray(np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(tmp)
    os.replace(tmp, path)               # never leaves a half-written keyframe behind


def _quantise(rgb):
    """The picture exactly as it is stored in the 8-bit PNG, so a later load gives the same pixels."""
    return (np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)).astype(np.float32) / 255.0


# ------------------------------------------------------------------ look and colour locks
def apply_look_lock(rgb, anchor_rgb, strength):
    """Keep a keyframe in the film's look: over-sharpened detail, hard edges, oversaturation and contrast that drifted
    past the gold frame's are pulled back (a chain of image edits slowly 'overcooks' a picture otherwise); lost detail
    gets only a gentle boost, because a soft shot can be soft on purpose."""
    h, w = rgb.shape[:2]
    pa = profile(np.clip(_resize(anchor_rgb, h, w), 0, 1))
    p0 = profile(rgb)
    if strength <= 0:
        rel = relative(p0, pa)
        return rgb, rel, rel
    k = float(strength)
    s = _settings(strength=k, texture=0.5, max_texture_gain=1.6, saturation_limit=1.12, saturation=1.0,
                  contrast=0.6, grain=0.0, texture_high=1.2, high_pull=1.0, edges_high=1.12, edges_pull=0.8)
    out = apply_correction(rgb, plan_correction(p0, pa, s), seed=0)
    # one band filter does not move the measured band by the full amount: tone down again until under the ceilings
    down = _settings(strength=k, texture=0.0, saturation=1.0, saturation_limit=1.12, contrast=0.0, grain=0.0,
                     texture_high=1.2, high_pull=1.0, edges_high=1.12, edges_pull=1.0)
    p = profile(out)
    for _ in range(3):
        r = relative(p, pa)
        if r["texture"] <= 1.2 * 1.03 and r["edges"] <= 1.12 * 1.03 and r["saturation"] <= 1.12 * 1.03:
            break
        out = apply_correction(out, plan_correction(p, pa, down), seed=0)
        p = profile(out)
    return out, relative(p0, pa), relative(p, pa)


def _colour_stats(lab):
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    mid = (L > 12.0) & (L < 92.0)
    neutral = mid & (np.hypot(a, b) < 14.0)
    use = neutral if neutral.mean() > 0.03 else mid
    if use.sum() < 64:
        use = np.ones_like(L, dtype=bool)
    return float(np.median(a[use])), float(np.median(b[use])), float(np.median(L)), bool(neutral.mean() > 0.03)


def _chroma(lab):
    L = lab[..., 0]
    mid = (L > 12.0) & (L < 92.0)
    c = np.hypot(lab[..., 1], lab[..., 2])
    return float(np.mean(c[mid])) if mid.sum() > 64 else float(np.mean(c))


def apply_colour_lock(rgb, base_rgb, amount, cast_free=6.0, cast_max=12.0, exp_free=6.0, exp_max=12.0):
    """Remove a small colour cast and exposure drift relative to the picture this one was made from.
    Differences up to cast_free / exp_free (Lab units) are drift and are corrected by `amount`; the correction fades
    out toward cast_max / exp_max; anything larger is taken as a deliberate change and left alone."""
    if _rgb_to_lab is None or amount <= 0:
        return rgb, None
    h, w = rgb.shape[:2]
    lab = _rgb_to_lab(rgb)
    blab = _rgb_to_lab(np.clip(_resize(base_rgb, h, w), 0, 1))
    a1, b1, L1, n1 = _colour_stats(lab)
    a0, b0, L0, n0 = _colour_stats(blab)
    da, db, dL = a1 - a0, b1 - b0, L1 - L0
    cast = math.hypot(da, db)

    def fade(v, free, mx):
        return 1.0 if v <= free else (0.0 if v >= mx else (mx - v) / (mx - free))

    # a deliberate lighting change (sunset, lights off) moves colour and brightness together: then nothing is touched
    deliberate = cast > cast_max or abs(dL) > exp_max
    kc = 0.0 if deliberate else float(amount) * fade(cast, cast_free, cast_max) * (1.0 if (n1 and n0) else 0.5)
    ke = 0.0 if deliberate else float(amount) * 0.6 * fade(abs(dL), exp_free, exp_max)
    if kc > 0:
        lab[..., 1] -= kc * da
        lab[..., 2] -= kc * db
    if ke > 0:
        lab[..., 0] = np.clip(lab[..., 0] - ke * dL, 0.0, 100.0)
    # saturation that crept up or down by up to 25% compared with the picture it was made from is drift too
    c1, c0 = _chroma(lab), _chroma(blab)
    ratio = c1 / max(c0, 1e-3)
    ks = 0.0 if (deliberate or not 0.75 <= ratio <= 1.33 or c0 < 2.0) else float(amount) * fade(abs(math.log(ratio)), 0.12, 0.29)
    if ks > 0:
        g = (1.0 / ratio) ** ks
        lab[..., 1] *= g
        lab[..., 2] *= g
    # contrast (spread of brightness) that crept by up to 25% is drift as well
    def spread(Lc):
        lo, hi = np.percentile(Lc, [5, 95])
        return float(hi - lo)
    sp1, sp0 = spread(lab[..., 0]), spread(blab[..., 0])
    cr = sp1 / max(sp0, 1e-3)
    kk = 0.0 if (deliberate or not 0.75 <= cr <= 1.33 or sp0 < 5.0) else float(amount) * fade(abs(math.log(cr)), 0.12, 0.29)
    if kk > 0:
        Lm = float(np.median(lab[..., 0]))
        lab[..., 0] = np.clip(Lm + (lab[..., 0] - Lm) * (1.0 / cr) ** kk, 0.0, 100.0)
    out = np.clip(_lab_to_rgb(lab), 0.0, 1.0).astype(np.float32) if (kc > 0 or ke > 0 or ks > 0 or kk > 0) else rgb
    return out, {"cast": cast, "cast_fixed": kc * cast, "exposure": dL, "exposure_fixed": ke * dL,
                 "saturation": ratio, "saturation_fixed": ks, "contrast": cr, "contrast_fixed": kk,
                 "kept_cast": deliberate, "kept_exposure": deliberate}


# ------------------------------------------------------------------ movement between first and last frame
STILL_BELOW = 0.02      # under 2% of the picture changed: the video model has (almost) nothing to animate
BIG_ABOVE = 0.40        # over 40%: a different camera angle or a new picture; the video may morph or cut


def _motion_grid(rgb, gw=64):
    """Brightness on a coarse grid (about 20 px cells for 1280 wide), normalised so that exposure or contrast drift
    does not count as movement."""
    h, w = rgb.shape[:2]
    gh = max(8, int(round(gw * h / float(w))))
    y = _resize(np.clip(rgb, 0, 1), gh * 4, gw * 4)
    lum = (0.299 * y[..., 0] + 0.587 * y[..., 1] + 0.114 * y[..., 2]).reshape(gh, 4, gw, 4).mean(axis=(1, 3))
    # light blur (3x3) so single noisy cells do not count
    p = np.pad(lum, 1, mode="edge")
    lum = sum(p[1 + dy:1 + dy + gh, 1 + dx:1 + dx + gw] * wt for dy, dx, wt in
              ((0, 0, 0.36), (-1, 0, 0.12), (1, 0, 0.12), (0, -1, 0.12), (0, 1, 0.12),
               (-1, -1, 0.04), (-1, 1, 0.04), (1, -1, 0.04), (1, 1, 0.04)))
    return (lum - float(np.median(lum))) / (float(lum.std()) + 1e-4)


def frame_change(img, base):
    """Share of the picture (0-1) that really changed between base and img. Each cell may match its neighbours, so
    the small shifts and redrawn texture of a regenerated picture do not count; people or objects that moved, turned
    or appeared do."""
    a = _motion_grid(np.asarray(img, dtype=np.float32))
    b = _motion_grid(np.asarray(base, dtype=np.float32))
    if b.shape != a.shape:
        b = _resize(b[..., None], a.shape[0], a.shape[1])[..., 0]
    H, W = a.shape
    p = np.pad(b, 1, mode="edge")
    best = np.full(a.shape, np.inf, dtype=np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            best = np.minimum(best, np.abs(a - p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]))
    return float(np.mean(best > 0.35))


def motion_word(ch):
    if ch < STILL_BELOW:
        return "STILL"
    if ch > BIG_ABOVE:
        return "BIG"
    return "ok"


def _first(t):
    return np.asarray(t[0].detach().cpu().numpy() if hasattr(t, "detach") else t[0], dtype=np.float32)[..., :3]


# ------------------------------------------------------------------ node
class StoryKeyframe:
    """First or last frame of a scene for the Story builder: made (checked and saved) or loaded."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "film_scene_01_first",
                                    "tooltip": "File name in ComfyUI/output/story (without .png)."}),
                "mode": (MODES, {"tooltip": "make: the image model makes the picture and it is saved. load: use the "
                                            "saved picture (stops if it is missing). load, or make if missing: the "
                                            "image model only runs when no picture is saved yet."}),
                "look_lock": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                                        "tooltip": "Pull detail, saturation and contrast back toward the film's "
                                                   "gold frame when they drifted. 0 = off."}),
                "colour_lock": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05,
                                          "tooltip": "Remove a small colour cast, or a small drift in brightness, "
                                                     "saturation or contrast, compared with the picture this one was "
                                                     "made from. Big, deliberate changes (lights off, sunset) are left "
                                                     "alone. 0 = off."}),
                "write_log": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image": ("IMAGE", {"lazy": True, "tooltip": "The new picture from the image model. Only made when "
                                                             "this node needs it."}),
                "anchor": ("IMAGE", {"lazy": True, "tooltip": "The film's gold frame (Quality Anchor)."}),
                "base": ("IMAGE", {"lazy": True, "tooltip": "The picture this keyframe was made from: the scene's "
                                                            "first frame, or the previous keyframe."}),
                "image_2": ("IMAGE", {"lazy": True, "tooltip": "Last frames only: a second picture with a stronger "
                                                               "change. Made only when 'image' is almost the same as "
                                                               "the first frame (the video would hardly move)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "video/ripcurlsurf"

    @classmethod
    def IS_CHANGED(cls, name="", mode="", **kw):
        # make: normal caching (a constant here, so a saved file never forces the video after it to render again).
        # load modes: the file time, so a picture you replaced by hand is used on the next run.
        if str(mode) == "make":
            return "make"
        p = _paths(name)[1]
        return os.path.getmtime(p) if os.path.isfile(p) else "missing"

    @staticmethod
    def _makes(name, mode):
        mode = str(mode)
        return mode == "make" or (mode.startswith("load, or make") and not os.path.isfile(_paths(name)[1]))

    @staticmethod
    def _is_last(name):
        return str(name or "").rstrip().lower().endswith("_last")

    def check_lazy_status(self, **kw):
        """ComfyUI asks this first: the picture (and its gold and base) are only needed when this node makes it.
        Once the first picture is there, the second one is asked for only when the first barely moved."""
        try:
            if not self._makes(kw.get("name", ""), kw.get("mode", "")):
                return []
            need = [n for n in ("image", "anchor", "base") if n in kw and kw[n] is None]
            if need:
                return need
            if ("image_2" in kw and kw["image_2"] is None and kw.get("image") is not None and kw.get("base") is not None
                    and self._is_last(kw.get("name", ""))):
                ch = frame_change(_first(kw["image"]), _first(kw["base"]))
                if ch < STILL_BELOW:
                    print("[StoryKeyframe] '%s': only %.1f%% of the picture changed from the first frame; making a "
                          "second picture with a stronger change" % (kw.get("name", ""), 100 * ch))
                    return ["image_2"]
            return []
        except Exception as e:
            print("[StoryKeyframe] check skipped (%s)" % e)
            return []

    def run(self, name, mode, look_lock=0.8, colour_lock=0.85, write_log=True, image=None, anchor=None, base=None,
            image_2=None):
        look, colour = float(look_lock), float(colour_lock)
        d, path, safe = _paths(name)
        ui = {"images": [{"filename": safe + ".png", "subfolder": _SUB, "type": "output"}]}
        if not self._makes(name, mode):
            if not os.path.isfile(path):
                rep = ("Story Keyframe '%s': output/story/%s.png is not there yet. Make this scene's keyframes first "
                       "(Story builder, Render = 'Keyframes only' or 'Keyframes and video' for this scene), or put your "
                       "own picture there with that name." % (safe, safe))
                _log({"node": "StoryKeyframe", "label": safe, "frames": 1, "verdict": "MISSING", "notes": rep}, rep,
                     write_log)
                raise RuntimeError(rep)
            try:
                arr = _load_png(path)
            except Exception as e:
                raise RuntimeError("Story Keyframe '%s': output/story/%s.png cannot be read (%s). Delete it and make "
                                   "the keyframe again." % (safe, safe, e))
            rep = "Story Keyframe '%s': LOADED output/story/%s.png (%dx%d, saved %s)" % (
                safe, safe, arr.shape[1], arr.shape[0],
                time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path))))
            _log({"node": "StoryKeyframe", "label": safe, "frames": 1, "verdict": "LOADED", "notes": rep}, rep,
                 write_log)
            return {"ui": ui, "result": (_to_tensor(arr[None]), rep)}

        if image is None:
            raise RuntimeError("Story Keyframe '%s': nothing is connected to 'image', so there is nothing to make. "
                               "Connect the image model's picture, or set the mode to 'load'." % safe)
        im = _Checked(image, "image")
        f = im.frame(0)
        notes = im.summary()
        if im.n > 1:
            notes.append("%d pictures came in; the first is used" % im.n)
        # last frame: how much moved since the first frame (a near-copy makes a frozen clip)
        mv, mv_note = None, ""
        base_f = None
        if base is not None:
            try:
                base_f = _Checked(base, "base").frame(0)
            except Exception as e:
                notes.append("base picture unusable: %s" % e)
        if base_f is not None and self._is_last(name):
            try:
                mv = frame_change(f, base_f)
                if image_2 is not None:
                    f2 = _Checked(image_2, "image_2").frame(0)
                    mv2 = frame_change(f2, base_f)
                    w1, w2 = motion_word(mv), motion_word(mv2)
                    use2 = (w2 == "ok") or (w1 == "STILL" and w2 == "STILL" and mv2 > mv)
                    mv_note = ("first picture: %.1f%% of the picture changed (%s); second picture: %.1f%% (%s); the %s "
                               "is used" % (100 * mv, w1, 100 * mv2, w2, "second" if use2 else "first"))
                    if use2:
                        f, mv = f2, mv2
            except Exception as e:
                notes.append("movement check skipped: %s" % e)
        if _is_blank(f):
            notes.append("WARNING: the new keyframe is blank (one flat colour): check the image model")
        rel_b = rel_a = None
        if anchor is not None and look > 0:
            try:
                g = _Checked(anchor, "anchor").frame(0)
                f, rel_b, rel_a = apply_look_lock(f, g, look)
            except Exception as e:
                notes.append("look lock skipped: %s" % e)
        cinfo = None
        if base_f is not None and colour > 0:
            try:
                f, cinfo = apply_colour_lock(f, base_f, colour)
            except Exception as e:
                notes.append("colour lock skipped: %s" % e)
        f = _quantise(f)
        try:
            _save_png(f, path)
            saved = "saved output/story/%s.png" % safe
        except Exception as e:
            saved = "NOT SAVED (%s): a later load of this keyframe will fail" % e
        parts = ["Story Keyframe '%s': MADE" % safe, saved]
        if rel_b is not None:
            parts.append("look vs the gold frame: quality %d -> %d (%s)" % (
                round(quality_score(rel_b, use_grain=False)), round(quality_score(rel_a, use_grain=False)), _fmt(rel_a)))
            fl = quality_flags(rel_a, use_grain=False)
            if fl:
                notes += fl
        if cinfo:
            if cinfo["kept_cast"] or cinfo["kept_exposure"]:
                notes.append("big colour or brightness change compared with the picture it was made from was kept "
                             "(taken as deliberate)")
            parts.append("colour cast %.1f, corrected %.1f; brightness change %+.1f, corrected %+.1f (Lab units); "
                         "saturation %d%% of the picture it was made from%s" % (
                             cinfo["cast"], cinfo["cast_fixed"], cinfo["exposure"], cinfo["exposure_fixed"],
                             round(100 * cinfo["saturation"]), ", corrected" if cinfo["saturation_fixed"] > 0 else ""))
        if mv is not None:
            w = motion_word(mv)
            parts.append("movement from the first frame: %.1f%% of the picture changed%s" % (100 * mv, {
                "STILL": " - STILL: almost the same picture, so the video will hardly move (the clip looks frozen). "
                         "Describe a clearer end of the action in this scene's Last frame box, or set Last frame to "
                         "'No last frame'",
                "BIG": " - BIG: a very different picture (another camera angle?); the video may morph or cut. Keep "
                       "the camera and place in the Last frame description",
                "ok": " (good: the video has something to animate)"}[w]))
            if mv_note:
                notes.append(mv_note)
        rep = " | ".join(parts) + ((" | " + "; ".join(notes)) if notes else "")
        verdict = "CHECK" if any(n.startswith("WARNING") for n in notes) else "MADE"
        if mv is not None and motion_word(mv) != "ok" and verdict == "MADE":
            verdict = motion_word(mv)
        row = {"node": "StoryKeyframe", "label": safe, "frames": 1, "verdict": verdict, "notes": rep}
        if rel_b is not None:
            row.update({"score_before": round(quality_score(rel_b, use_grain=False), 1),
                        "score_after": round(quality_score(rel_a, use_grain=False), 1),
                        "texture_before": round(rel_b["texture"], 3), "texture_after": round(rel_a["texture"], 3),
                        "saturation_before": round(rel_b["saturation"], 3),
                        "saturation_after": round(rel_a["saturation"], 3),
                        "contrast_before": round(rel_b["contrast"], 3), "contrast_after": round(rel_a["contrast"], 3)})
        _log(row, rep, write_log)
        return {"ui": ui, "result": (_to_tensor(f[None]), rep)}


NODE_CLASS_MAPPINGS = {"StoryKeyframe": StoryKeyframe}
NODE_DISPLAY_NAME_MAPPINGS = {"StoryKeyframe": "Story Keyframe (first or last frame of a scene, saved)"}
