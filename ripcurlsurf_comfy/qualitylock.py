"""
qualitylock.py - ComfyUI custom nodes that stop picture quality dropping from scene to scene
(package: ripcurlsurf_comfy).

WHY QUALITY DROPS
Every scene that continues from the previous one starts from that scene's last frame. Small losses in that
frame (softness, smoothed skin and fabric, rising saturation, harder edges, missing grain) are copied into the
next scene, which adds its own. After a few scenes the film looks painted or like a cartoon. Inside one long
clip the same slide happens between the first frame and the last.

WHAT THESE NODES DO
They measure frames against one fixed GOLD frame (the opening picture of the film) and correct the difference,
inside every scene and at every join:

  Quality Anchor (gold frame)          stores the gold frame on disk the first time and loads it in later runs,
                                       so batches and single-scene redos all aim at the same target
  Scene Quality Lock (whole clip)      corrects every frame of a scene toward the gold frame's texture, grain,
                                       saturation and contrast. The correction is smoothed over time so it never
                                       flickers, and it never moves, adds or removes anything in the picture
  Best Hand-off Frame                  picks the sharpest of the last few frames for the next scene to start
                                       from (the very last frame of a clip is often the softest)
  Quality Lock (one frame)             the same correction for a single picture
  Quality Gate (pick best candidate)   compares candidate hand-off frames (the corrected original and image-model
                                       restores), rejects any that changed the layout or the detail of the picture,
                                       and keeps the one the next clip should start from. A soft hand-off counts
                                       five times as much as a slightly crisp one (the next clip only ever loses
                                       detail), and a real restore wins over the original whose detail was only
                                       boosted. The stronger restores (candidates 3 and 4) are lazy: ComfyUI makes
                                       them only when the best restore so far is still soft

WHAT IS MEASURED (relative to the gold frame: 1.00 = same as gold)
  texture     fine detail relative to medium detail (the spectral slope). Drops when skin, hair and fabric are
              smoothed: the main sign of the painted or cartoon look
  edges       medium detail relative to large shapes. Rises when outlines become hard and graphic
  grain       fine noise in the flat parts of the picture. Real footage has grain; a too-clean frame reads as CG
  saturation  average colourfulness of the midtones
  contrast    spread of brightness (5th to 95th percentile)
A quality score (0-100) combines them. Every node validates its input (shape, empty batch, NaN or infinite
values, values outside 0-1, blank frames) and writes one line per run to ComfyUI/output/movie_log.txt, the one log
file every ripcurlsurf node writes to. Scenes that stay below the limits after correction are marked RE-RENDER there.

Plain numpy, plus Pillow (already part of ComfyUI) when available. No model, no extra installs.
"""
import math
import os
import re
import time

import numpy as np
import torch

VERSION = "2026.10.20.1"
_EPS = 1e-6
_NOISE_FLOOR = 0.0015          # about the 8-bit quantisation level; keeps grain ratios meaningful
_RERENDER_SCORE = 55.0         # scene still below this after correction -> RE-RENDER
_CHECK_SCORE = 70.0            # below this -> CHECK


# ------------------------------------------------------------------ small filters (numpy only)
def _sl(axis, ndim, s):
    idx = [slice(None)] * ndim
    idx[axis] = s
    return tuple(idx)


def _box_axis(a, r, axis):
    """Box filter of radius r along one axis (running sum, reflect padding). Cost does not depend on r."""
    n = a.shape[axis]
    if r <= 0 or n < 3:
        return a
    r = int(min(r, n - 2))
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    p = np.pad(a, pad, mode="reflect").astype(np.float64)
    m = p.mean(axis=axis, keepdims=True)
    c = np.cumsum(p - m, axis=axis)
    w = 2 * r + 1
    out = (c[_sl(axis, c.ndim, slice(w, w + n))] - c[_sl(axis, c.ndim, slice(0, n))]) / w + m
    return out.astype(np.float32)


def _binom(a):
    """3-tap binomial blur [1 2 1]/4 on both axes (sigma ~0.7 pixel)."""
    out = a
    for axis in (0, 1):
        n = out.shape[axis]
        if n < 3:
            continue
        pad = [(0, 0)] * out.ndim
        pad[axis] = (1, 1)
        p = np.pad(out, pad, mode="reflect")
        out = (0.25 * p[_sl(axis, p.ndim, slice(0, n))] + 0.5 * p[_sl(axis, p.ndim, slice(1, n + 1))]
               + 0.25 * p[_sl(axis, p.ndim, slice(2, n + 2))])
    return out.astype(np.float32, copy=False)


def _gauss(a, sigma):
    """Gaussian blur approximated by three box passes on axes 0 and 1."""
    if sigma < 0.3:
        return a
    if sigma < 0.6:
        return _binom(a)
    r = max(1, int(round((math.sqrt(4.0 * sigma * sigma + 1.0) - 1.0) / 2.0)))
    out = a
    for _ in range(3):
        out = _box_axis(out, r, 0)
        out = _box_axis(out, r, 1)
    return out


def _luma(rgb):
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)


def _resize(img, h, w):
    """Resize [H,W] or [H,W,C] float32 to h x w (Pillow Lanczos per channel, numpy fallback)."""
    h, w = int(h), int(w)
    if img.shape[0] == h and img.shape[1] == w:
        return img
    try:
        from PIL import Image
        rs = getattr(Image, "Resampling", Image).LANCZOS
        chans = [img] if img.ndim == 2 else [img[..., k] for k in range(img.shape[2])]
        out = [np.asarray(Image.fromarray(np.ascontiguousarray(c, dtype=np.float32)).resize((w, h), rs),
                          dtype=np.float32) for c in chans]
        return out[0] if img.ndim == 2 else np.stack(out, axis=-1)
    except Exception:
        src = img.astype(np.float32)
        sh, sw = src.shape[0] / float(h), src.shape[1] / float(w)
        if sh > 1.5 or sw > 1.5:                       # pre-blur so a big downscale does not alias
            src = _gauss(src, 0.5 * max(sh, sw))
        ys = np.clip((np.arange(h) + 0.5) * sh - 0.5, 0, src.shape[0] - 1)
        xs = np.clip((np.arange(w) + 0.5) * sw - 0.5, 0, src.shape[1] - 1)
        y0 = np.floor(ys).astype(int)
        x0 = np.floor(xs).astype(int)
        y1 = np.minimum(y0 + 1, src.shape[0] - 1)
        x1 = np.minimum(x0 + 1, src.shape[1] - 1)
        wy = (ys - y0).reshape(-1, 1)
        wx = (xs - x0).reshape(1, -1)
        if src.ndim == 3:
            wy, wx = wy[..., None], wx[..., None]
        top = src[y0][:, x0] * (1 - wx) + src[y0][:, x1] * wx
        bot = src[y1][:, x0] * (1 - wx) + src[y1][:, x1] * wx
        return (top * (1 - wy) + bot * wy).astype(np.float32)


def _inner(a, frac=0.04):
    h, w = a.shape[0], a.shape[1]
    dy, dx = int(h * frac), int(w * frac)
    return a[dy:h - dy if dy else None, dx:w - dx if dx else None]


# ------------------------------------------------------------------ validation
class _Checked:
    """Turns a ComfyUI IMAGE input into numpy and records every problem found."""

    def __init__(self, x, name):
        if x is None:
            raise ValueError("%s: nothing is connected" % name)
        a = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        if a.ndim == 3:
            a = a[None]
        if a.ndim != 4:
            raise ValueError("%s: expected [frames, height, width, channels], got shape %s" % (name, tuple(a.shape)))
        if a.shape[0] == 0:
            raise ValueError("%s: the batch is empty (0 frames)" % name)
        if a.shape[1] < 16 or a.shape[2] < 16:
            raise ValueError("%s: the picture is only %dx%d pixels" % (name, a.shape[2], a.shape[1]))
        if a.shape[3] < 1:
            raise ValueError("%s: the picture has no colour channels" % name)
        self.name = name
        self.raw = a
        self.n, self.h, self.w = a.shape[0], a.shape[1], a.shape[2]
        self.channels = a.shape[3]
        self.issues = []
        self.nonfinite = 0
        self.out_of_range = 0
        if self.channels == 2:
            self.issues.append("%s has 2 channels; the first is used as grey" % name)
        elif self.channels > 4:
            self.issues.append("%s has %d channels; the first 3 are used" % (name, self.channels))

    def alpha(self, i):
        return self.raw[i, ..., 3:4].astype(np.float32) if self.channels == 4 else None

    def frame(self, i):
        """One frame as clean float32 RGB [H,W,3] in 0..1 (problems are counted, never passed on)."""
        f = self.raw[i]
        if self.channels < 3:
            f = np.repeat(f[..., :1], 3, axis=-1)
        else:
            f = f[..., :3]
        f = np.array(f, dtype=np.float32, copy=True)
        fin = np.isfinite(f)
        if not fin.all():
            self.nonfinite += int((~fin).sum())
            f = np.nan_to_num(f, nan=0.0, posinf=1.0, neginf=0.0)
        lo, hi = float(f.min()), float(f.max())
        if lo < -1e-3 or hi > 1.0 + 1e-3:
            self.out_of_range += 1
        return np.clip(f, 0.0, 1.0)

    def summary(self):
        s = list(self.issues)
        if self.nonfinite:
            s.append("%s contained %d NaN/infinite values (replaced)" % (self.name, self.nonfinite))
        if self.out_of_range:
            s.append("%s had %d frame(s) with values outside 0-1 (clipped)" % (self.name, self.out_of_range))
        return s


def _is_blank(rgb):
    y = _luma(rgb[::4, ::4])
    return float(y.std()) < 0.004


# ------------------------------------------------------------------ measuring
def _bands(y):
    b1 = _binom(y)
    g2 = _gauss(y, 2.5)
    g3 = _gauss(y, 7.0)
    return y - b1, b1 - g2, g2 - g3


def profile(rgb):
    """Quality profile of one RGB frame [H,W,3] in 0..1."""
    y = _luma(rgb)
    fine, mid, coarse = _bands(y)
    fi, mi, co = _inner(fine)[::2, ::2], _inner(mid)[::2, ::2], _inner(coarse)[::2, ::2]
    ef = float(np.sqrt(np.mean(fi * fi))) + _EPS
    em = float(np.sqrt(np.mean(mi * mi))) + _EPS
    ec = float(np.sqrt(np.mean(co * co))) + _EPS
    act = _box_axis(_box_axis(np.abs(mid), 2, 0), 2, 1)          # local amount of medium detail
    ai = _inner(act)[::2, ::2]
    flat = ai <= np.percentile(ai, 40)                              # the flattest 40 percent
    sel = np.abs(fi[flat]) if flat.sum() > 64 else np.abs(fi).ravel()
    noise = max(_NOISE_FLOOR, 1.4826 * float(np.median(sel)))
    ys = _inner(y)[::3, ::3]
    cs = _inner(rgb)[::3, ::3]
    mm = (ys > 0.12) & (ys < 0.92)
    dev = np.abs(cs - ys[..., None]).mean(axis=-1)
    sat = float(dev[mm].mean()) if mm.sum() > 50 else float(dev.mean())
    p5, p95 = np.percentile(ys, [5, 95])
    # SURFACE TEXTURE: medium-fine detail (skin pores, hair, fabric weave) on the surfaces between the big shapes,
    # as an absolute amount (scaled by the picture's contrast). Unlike the ratios below it drops when a frame turns
    # waxy at every scale at once - the plastic look - and grain hardly moves it (grain is finer than this band).
    cact = _box_axis(_box_axis(np.abs(coarse), 3, 0), 3, 1)
    ca = _inner(cact)[::2, ::2]
    surf_m = ca <= np.percentile(ca, 60)
    sm = mi[surf_m] if surf_m.sum() > 64 else mi.ravel()
    con = float(p95 - p5) + 1e-3
    surface = float(np.sqrt(np.mean(sm * sm))) / con + _EPS
    return {"surface": surface, "ef": ef, "em": em, "ec": ec, "rf": ef / em, "rm": em / ec, "noise": noise,
            "sat": max(sat, 1e-4), "con": float(p95 - p5) + 1e-3, "mean": float(ys.mean()),
            "clip": float(((ys > 0.985) | (ys < 0.015)).mean()), "blank": float(ys.std()) < 0.004}


def relative(p, a):
    return {"texture": p["rf"] / a["rf"], "edges": p["rm"] / a["rm"], "grain": p["noise"] / a["noise"],
            "saturation": p["sat"] / a["sat"], "contrast": p["con"] / a["con"], "clip": p["clip"],
            "surface": p["surface"] / a["surface"]}


def _ln(v):
    return math.log(max(float(v), 1e-3))


_SURFACE_LOW = 0.85            # surface texture below 85% of the target counts as waxy (the plastic look)


def _surface_term(rel, target=1.0):
    sv = max(float(rel.get("surface", 1.0)), 1e-3) / max(float(target), 1e-3)
    return 2.0 * max(0.0, math.log(_SURFACE_LOW) - math.log(sv)) + 0.3 * max(0.0, math.log(sv) - math.log(1.6))


def quality_distance(rel, use_grain=True):
    d = abs(_ln(rel["texture"]))
    if use_grain:
        d += 0.3 * abs(_ln(rel["grain"]))
    d += 1.6 * max(0.0, _ln(rel["saturation"]) - math.log(1.12)) + 0.6 * max(0.0, math.log(0.75) - _ln(rel["saturation"]))
    d += 0.5 * max(0.0, abs(_ln(rel["contrast"])) - math.log(1.08))
    d += 0.5 * max(0.0, _ln(rel["edges"]) - math.log(1.15))
    d += 4.0 * max(0.0, rel["clip"] - 0.03)
    return d


def quality_score(rel, use_grain=True):
    return 100.0 * math.exp(-quality_distance(rel, use_grain))


_HANDOFF_CEIL = 1.15           # detail up to 15% above the target is fine for a hand-off frame
_HANDOFF_HIGH = 1.12           # the gate tones restores down to this (crunchy detail would start the next clip)
_FALLBACK_MARGIN = 0.15        # the boosted original must beat the best restore by this much to be used


def handoff_distance(rel, target=1.0, edges_target=1.0, surface_target=1.0):
    """How far a hand-off frame is from what the next clip should start from (0 = ideal).
    Missing detail weighs about five times as much as extra detail up to 15% above the target: the next clip
    starts from this frame and only ever loses detail, so a soft start compounds scene after scene, while a
    slightly crisp one settles back within the clip. Edges harder than 8% above the shot's edge target weigh double
    here: carried from join to join they grow into the hard, graphic cartoon look.
    Surface texture (the waxy/plastic measure) counts in full; grain counts a little, so a too-clean frame loses."""
    t = max(float(rel["texture"]), 1e-3) / max(float(target), 1e-3)
    lt = math.log(t)
    if lt < 0.0:
        d = 1.6 * -lt
    elif t <= _HANDOFF_CEIL:
        d = 0.3 * lt
    else:
        d = 0.3 * math.log(_HANDOFF_CEIL) + 1.2 * (lt - math.log(_HANDOFF_CEIL))
    d += 1.6 * max(0.0, _ln(rel["saturation"]) - math.log(1.12)) + 0.6 * max(0.0, math.log(0.75) - _ln(rel["saturation"]))
    d += 0.5 * max(0.0, abs(_ln(rel["contrast"])) - math.log(1.08))
    d += 1.0 * max(0.0, _ln(rel["edges"]) - math.log(1.08 * max(float(edges_target), 1e-3)))
    d += 4.0 * max(0.0, rel["clip"] - 0.03)
    d += 0.25 * max(0.0, math.log(0.6) - _ln(rel.get("grain", 1.0)))
    d += _surface_term(rel, surface_target)       # the waxy/plastic measure (was defined but never counted)
    return d


def quality_flags(rel, use_grain=True):
    f = []
    t = rel["texture"]
    if t < 0.65:
        f.append("painted look (texture %d%% of gold)" % round(100 * t))
    elif t < 0.8:
        f.append("soft (texture %d%% of gold)" % round(100 * t))
    elif t > 1.35:
        f.append("over-sharpened (texture %d%% of gold)" % round(100 * t))
    if rel["saturation"] > 1.22:
        f.append("oversaturated (%d%% of gold)" % round(100 * rel["saturation"]))
    elif rel["saturation"] < 0.7:
        f.append("washed out (%d%% of gold)" % round(100 * rel["saturation"]))
    if use_grain and rel["grain"] < 0.5:
        f.append("too clean (grain %d%% of gold)" % round(100 * rel["grain"]))
    elif use_grain and rel["grain"] > 2.2:
        f.append("noisy (grain %d%% of gold)" % round(100 * rel["grain"]))
    if rel["edges"] > 1.3:
        f.append("hard graphic edges")
    if rel["contrast"] > 1.3:
        f.append("contrast too high")
    elif rel["contrast"] < 0.75:
        f.append("flat contrast")
    if rel["clip"] > 0.08:
        f.append("clipped highlights or shadows (%d%%)" % round(100 * rel["clip"]))
    return f


def _fmt(rel):
    return ("surface %.2f, " % rel["surface"] if "surface" in rel else "") + "texture %.2f, grain %.2f, saturation %.2f, contrast %.2f, edges %.2f" % (
        rel["texture"], rel["grain"], rel["saturation"], rel["contrast"], rel["edges"])


# ------------------------------------------------------------------ correcting
def _settings(strength=0.8, texture=1.0, max_texture_gain=2.2, saturation_limit=1.15, saturation=1.0,
              contrast=0.5, grain=1.0, coring=0.35, texture_high=1.3, high_pull=None, edges_high=1.15, edges_pull=None):
    """texture_high: detail above this multiple of the gold frame is toned down (1.3 for footage: only real
    over-sharpening). high_pull: how strongly (None = same as the boost). edges_high / edges_pull: the same for
    hard medium-size edges (the graphic, cartoon-like look)."""
    c01 = lambda v: float(min(1.0, max(0.0, v)))
    return {"k": c01(strength), "texture": c01(texture), "max_gain": float(min(3.0, max(1.0, max_texture_gain))),
            "sat_limit": float(min(2.0, max(1.0, saturation_limit))), "saturation": c01(saturation),
            "contrast": c01(contrast), "grain": float(min(2.0, max(0.0, grain))), "coring": float(max(0.0, coring)),
            "hi": float(max(1.0, texture_high)), "hi_pull": None if high_pull is None else c01(high_pull),
            "ehi": float(max(0.85, edges_high)), "epull": None if edges_pull is None else c01(edges_pull)}


def plan_correction(p, a, s):
    """Correction parameters that move profile p toward gold profile a (all clamped)."""
    k = s["k"]
    rel = relative(p, a)
    # Dead bands: natural differences between shots (a clear sky, dense foliage) are left alone; only real
    # drift outside the band is corrected, and only back to the edge of the band.
    e = max(rel["edges"], 1e-3)
    ehi = s.get("ehi", 1.15)
    if e > ehi:
        gm = (ehi / e) ** (0.5 * s["texture"] * k if s.get("epull") is None else s["epull"])
    elif e < 0.8:
        gm = (1.0 / e) ** (0.5 * s["texture"] * k)
    else:
        gm = 1.0
    gm = float(np.clip(gm, 0.85, 1.2))
    t = max(rel["texture"], 1e-3) / gm          # texture once the medium detail has been rescaled
    hi = s.get("hi", 1.3)
    if t < 0.95:
        # only a light lift of the finest band: sharpening it hard over waxy surfaces is what made frames look plastic
        # (crisp outlines on smooth skin). Lost surface texture is handled by the surface gain below.
        gf = min(1.15, (1.0 / t) ** (0.5 * s["texture"] * k))
    elif t > hi:
        gf = (hi / t) ** (s["texture"] * k if s.get("hi_pull") is None else s["hi_pull"])
    else:
        gf = 1.0
    gf = float(np.clip(gf, 0.8, s["max_gain"]))
    sr = rel["saturation"]
    if sr > s["sat_limit"]:
        sc = (s["sat_limit"] / sr) ** (s["saturation"] * k)
    elif sr < 0.7:
        sc = min(1.25, (0.85 / sr) ** (0.5 * s["saturation"] * k))
    else:
        sc = 1.0
    cr = max(rel["contrast"], 1e-3)
    gc = (1.08 / cr) ** (s["contrast"] * k) if cr > 1.08 else ((0.92 / cr) ** (s["contrast"] * k) if cr < 0.92 else 1.0)
    gc = float(np.clip(gc, 0.85, 1.15))
    # surface gain: lift medium-fine detail on the surfaces between big shapes (skin, fabric, walls), never on edges
    sv = max(float(rel.get("surface", 1.0)), 1e-3)
    gs = float(np.clip((0.95 / sv) ** (s["texture"] * k), 1.0, 1.6)) if sv < 0.95 else 1.0
    cur = p["noise"] * (gf if gf < 1.0 else 1.0)
    tgt = a["noise"] * s["grain"]
    add = k * math.sqrt(max(0.0, tgt * tgt - cur * cur))
    return {"gf": gf, "gm": gm, "gs": gs, "sc": float(sc), "gc": gc, "grain": float(min(add, 0.04)),
            "tau": s["coring"] * p["noise"]}


_GRAIN_UNIT = {}


def _grain_field(h, w, seed):
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    n = _binom(rng.standard_normal((h, w), dtype=np.float32))
    return n / (float(n.std()) + _EPS)


def _grain_unit():
    """How much the 'noise' measurement rises per unit of our grain field (measured once)."""
    if "u" not in _GRAIN_UNIT:
        g = _grain_field(256, 256, 12345)
        _GRAIN_UNIT["u"] = 1.4826 * float(np.median(np.abs(g - _binom(g)))) + _EPS
    return _GRAIN_UNIT["u"]


def apply_correction(rgb, c, seed=0):
    """Apply a correction plan to one RGB frame. Only tone, detail, saturation and grain change."""
    y = _luma(rgb)
    b1 = _binom(y)
    fine = y - b1
    dy = np.zeros_like(y)
    gf = c["gf"]
    if abs(gf - 1.0) > 0.01:
        dy += (gf - 1.0) * fine                                # no coring: coring kept edges and dropped pores
    g2 = _gauss(y, 2.5)
    mid = b1 - g2
    if abs(c["gm"] - 1.0) > 0.01:
        dy += (c["gm"] - 1.0) * mid
    gs = float(c.get("gs", 1.0))
    if gs > 1.01:
        # weight 1 on surfaces, 0 on the big edges (where medium detail is strong); edges are never hardened
        g3 = _gauss(y, 7.0)
        act = _gauss(np.abs(g2 - g3), 3.0)
        ref = float(np.percentile(act, 85)) + _EPS
        wsurf = np.clip(1.0 - act / ref, 0.0, 1.0) ** 1.5
        dy += (gs - 1.0) * (fine + mid) * wsurf
    out = rgb + dy[..., None]
    y2 = y + dy
    if abs(c["sc"] - 1.0) > 0.005:
        out = y2[..., None] + c["sc"] * (out - y2[..., None])
    if abs(c["gc"] - 1.0) > 0.005:
        out = out + ((c["gc"] - 1.0) * (y2 - float(y2.mean())))[..., None]
    if c["grain"] > 1e-5:
        lm = np.clip(y2, 0.0, 1.0)
        wmid = 0.35 + 0.65 * 4.0 * lm * (1.0 - lm)
        g = _grain_field(rgb.shape[0], rgb.shape[1], seed)
        out = out + ((c["grain"] / _grain_unit()) * g * wmid)[..., None]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _smooth(v, win):
    v = np.asarray(v, dtype=np.float64)
    if win <= 1 or v.size < 3:
        return v
    win = int(min(win, v.size if v.size % 2 else v.size - 1))
    if win < 3:
        return v
    if win % 2 == 0:
        win -= 1
    pad = win // 2
    p = np.pad(v, (pad, pad), mode="edge")
    return np.convolve(p, np.ones(win) / win, mode="valid")


# ------------------------------------------------------------------ similarity (for the gate)
def _ssim(x, y, sigma=1.5):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mx, my = _gauss(x, sigma), _gauss(y, sigma)
    sxx = _gauss(x * x, sigma) - mx * mx
    syy = _gauss(y * y, sigma) - my * my
    sxy = _gauss(x * y, sigma) - mx * my
    m = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return float(np.mean(m))


def layout_similarity(a, b):
    """SSIM of the two pictures shrunk to about 160 pixels: same shapes in the same places?"""
    ya, yb = _luma(a), _luma(b)
    h, w = ya.shape
    s = 160.0 / max(h, w)
    hh, ww = max(16, int(round(h * s))), max(16, int(round(w * s)))
    return _ssim(_resize(ya, hh, ww), _resize(yb, hh, ww), 1.5)


def detail_similarity(a, b):
    """Correlation of medium-size detail (eyes, mouths, outlines, lettering) at about 400 pixels."""
    ya, yb = _luma(a), _luma(b)
    h, w = ya.shape
    s = min(1.0, 400.0 / max(h, w))
    hh, ww = max(16, int(round(h * s))), max(16, int(round(w * s)))
    ra, rb = _resize(ya, hh, ww), _resize(yb, hh, ww)
    da, db = _gauss(ra, 1.0) - _gauss(ra, 3.0), _gauss(rb, 1.0) - _gauss(rb, 3.0)
    sab, saa, sbb = _gauss(da * db, 2.0), _gauss(da * da, 2.0), _gauss(db * db, 2.0)
    loc = (2 * sab + 1e-6) / (saa + sbb + 1e-6)
    wgt = np.sqrt(np.maximum(saa + sbb, 0.0))
    return float((loc * wgt).sum() / (wgt.sum() + _EPS))


# ------------------------------------------------------------------ shared helpers
def _out_dir():
    try:
        import folder_paths
        return folder_paths.get_output_directory()
    except Exception:
        return os.path.join(os.getcwd(), "output")


# ------------------------------------------------------------------ the one log file
# Every ripcurlsurf node appends to ComfyUI/output/movie_log.txt, in the order things happen:
#   2026-10-16 17:08:28 | Scene guard     | Scene 1 (file 01) | OK | planned 243 frames, got 243, ...
#   time                | section         | what              | verdict | details (more " | " parts)
# Troubleshooting entries add indented lines under their first line. Lines that start with a date are entries;
# the finishing script's quality report reads them.
LOG_FILE = "movie_log.txt"
_SECTION = {"StoryKeyframe": "Story picture", "QualityAnchor": "Gold frame", "SceneQualityLock": "Quality lock",
            "QualityLock": "Hand-off lock", "BestHandoffFrame": "Hand-off frame", "QualityGate": "Join check",
            "SceneGuard": "Scene guard", "TroubleshootClip": "Troubleshooting"}


def _one_line(s):
    return " / ".join(p.strip() for p in str(s).replace("\r", "\n").split("\n") if p.strip())


def _detail(label, verdict, text):
    """The details of an entry: the report without its leading 'label: VERDICT' (the line already shows both)."""
    t = _one_line(text)
    for pre in ("Story Keyframe ", "Quality Anchor: "):
        if t.startswith(pre):
            t = t[len(pre):]
            break
    for lab in ("'%s'" % label, str(label)):
        if lab and t.startswith(lab + ": "):
            t = t[len(lab) + 2:]
            break
    m = re.match(r"([A-Z][A-Z-]+)(?: \| |: | |$)", t)
    if m and (m.group(1) == verdict or m.group(1) == "MADE"):
        t = t[m.end():]
    return t.strip()


def movie_log(node, label, verdict, text, more=None, enabled=True):
    """Append one entry to ComfyUI/output/movie_log.txt. 'more' adds indented lines under it. Never raises."""
    if not enabled:
        return None
    try:
        d = _out_dir()
        os.makedirs(d, exist_ok=True)
        lab = _one_line(label).replace("|", "/") or "-"
        det = _detail(label, verdict, text)
        line = "%s | %-15s | %s | %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), _SECTION.get(node, node), lab,
                                         _one_line(verdict) or "-") + ((" | " + det) if det else "")
        out = [line]
        for m in more or []:
            out += ["    " + p.rstrip() for p in str(m).replace("\r", "").split("\n") if p.strip()]
        p = os.path.join(d, LOG_FILE)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write("\n".join(out) + "\n")
        return p
    except Exception as e:
        print("[ripcurlsurf_comfy] could not write output/%s: %s" % (LOG_FILE, e))
        return None


def _log(row, text, enabled=True):
    print("[QualityLock] " + text)
    movie_log(row.get("node", ""), row.get("label", ""), row.get("verdict", ""), text, enabled=enabled)


def _verdict(score_after, rel_after):
    if score_after < _RERENDER_SCORE or rel_after["texture"] < 0.7:
        return "RE-RENDER"
    if score_after < _CHECK_SCORE or quality_flags(rel_after):
        return "CHECK"
    return "OK"


def _gold(anchor_checked, h, w):
    g = anchor_checked.frame(0)
    g = _resize(g, h, w)
    pa = profile(g)
    notes = []
    if pa["blank"]:
        notes.append("the gold frame is blank: correction is weak")
    if pa["noise"] <= _NOISE_FLOOR * 1.01:
        notes.append("the gold frame has almost no grain")
    if pa["clip"] > 0.15:
        notes.append("the gold frame is %d%% clipped" % round(100 * pa["clip"]))
    return pa, notes


def _to_tensor(arr):
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))


def clip_quality(frames, anchor, samples=5):
    """Quality of a clip against a gold frame, used by Scene Guard to rank tries.
    frames: numpy [N,H,W,C] 0..1; anchor: numpy [H,W,C] or [1,H,W,C]. Returns a dict."""
    n, h, w = frames.shape[0], frames.shape[1], frames.shape[2]
    a = anchor[0] if anchor.ndim == 4 else anchor
    a = np.clip(np.nan_to_num(a[..., :3].astype(np.float32)), 0, 1)
    if a.shape[-1] < 3:
        a = np.repeat(a[..., :1], 3, axis=-1)
    pa = profile(_resize(a, h, w))
    idx = sorted(set(int(round(x)) for x in np.linspace(0, n - 1, max(2, min(samples, n)))))
    rels = []
    for i in idx:
        f = frames[i][..., :3] if frames.shape[-1] >= 3 else np.repeat(frames[i][..., :1], 3, axis=-1)
        f = np.clip(np.nan_to_num(f.astype(np.float32)), 0, 1)
        p = profile(f)
        if not p["blank"]:
            rels.append(relative(p, pa))
    if not rels:
        return {"score": 0.0, "penalty": 9.0, "flags": ["all sampled frames are blank"], "texture_start": 0.0,
                "texture_end": 0.0}
    dists = [quality_distance(r) for r in rels]
    drop = max(0.0, rels[0]["texture"] - rels[-1]["texture"])
    worst = rels[int(np.argmax(dists))]
    return {"score": float(np.mean([100.0 * math.exp(-d) for d in dists])),
            "texture_mean": float(np.mean([r["texture"] for r in rels])),
            "saturation_mean": float(np.mean([r["saturation"] for r in rels])),
            "penalty": float(np.mean(dists) + 1.5 * drop), "flags": quality_flags(worst),
            "texture_start": rels[0]["texture"], "texture_end": rels[-1]["texture"]}


# ------------------------------------------------------------------ nodes
_SETTINGS_INPUTS = [
    ("strength", ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                            "tooltip": "Overall amount of correction. 0 = measure and log only."})),
    ("texture", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                           "tooltip": "How strongly lost fine detail (skin, hair, fabric) is restored toward the gold frame."})),
    ("max_texture_gain", ("FLOAT", {"default": 2.2, "min": 1.0, "max": 3.0, "step": 0.05,
                                    "tooltip": "Upper limit for the detail boost, so noise is never blown up."})),
    ("saturation_limit", ("FLOAT", {"default": 1.15, "min": 1.0, "max": 2.0, "step": 0.01,
                                    "tooltip": "Saturation above this multiple of the gold frame is brought back down."})),
    ("saturation", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05})),
    ("contrast", ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05})),
    ("grain", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                         "tooltip": "Grain is added until it matches the gold frame times this value."})),
    ("grain_seed", ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF})),
    ("write_log", ("BOOLEAN", {"default": True})),
]


class SceneQualityLock:
    """Whole clip: measure every scene against the gold frame and correct it, smoothly over time."""

    @classmethod
    def INPUT_TYPES(cls):
        req = {"images": ("IMAGE",),
               "anchor": ("IMAGE", {"tooltip": "The gold frame (output of Quality Anchor)."}),
               "label": ("STRING", {"default": "Scene"}),
               "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01})}
        req.update(dict(_SETTINGS_INPUTS))
        return {"required": req}

    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT")
    RETURN_NAMES = ("images", "report", "quality")
    FUNCTION = "run"
    CATEGORY = "video/ripcurlsurf"

    def run(self, images, anchor, label, fps, strength, texture, max_texture_gain, saturation_limit, saturation,
            contrast, grain, grain_seed, write_log):
        t0 = time.time()
        clip = _Checked(images, "images")
        gold = _Checked(anchor, "anchor")
        n, h, w = clip.n, clip.h, clip.w
        fps = float(fps) if fps and fps > 0 else 24.0
        s = _settings(strength, texture, max_texture_gain, saturation_limit, saturation, contrast, grain)
        pa, notes = _gold(gold, h, w)
        notes = gold.summary() + notes
        stride = max(1, int(round(fps / 6.0)))
        idx = sorted(set(list(range(0, n, stride)) + [n - 1]))
        plans, blanks, rels_before = {}, [], {}
        for i in idx:
            f = clip.frame(i)
            p = profile(f)
            if p["blank"]:
                blanks.append(i)
                continue
            plans[i] = plan_correction(p, pa, s)
            rels_before[i] = relative(p, pa)
        if not plans:
            rep = "%s: CHECK | every sampled frame is blank - nothing to correct" % label
            _log({"node": "SceneQualityLock", "label": label, "frames": n, "verdict": "CHECK", "notes": rep}, rep,
                 write_log)
            return (images, rep, 0.0)
        keys = sorted(plans)
        win = max(1, int(round(fps)))
        series = {}
        for name in ("gf", "gm", "gs", "sc", "gc", "grain", "tau"):
            v = np.interp(np.arange(n), keys, [plans[k][name] for k in keys])
            series[name] = _smooth(v, win)
        out = np.empty((n, h, w, 4 if clip.channels == 4 else 3), dtype=np.float32)
        nblank = 0
        for i in range(n):
            f = clip.frame(i)
            if _is_blank(f):
                nblank += 1
                g = f
            else:
                c = {name: float(series[name][i]) for name in series}
                g = apply_correction(f, c, seed=int(grain_seed) + 7919 * i)
            if clip.channels == 4:
                out[i] = np.concatenate([g, clip.alpha(i)], axis=-1)
            else:
                out[i] = g
        notes += clip.summary()
        if nblank:
            notes.append("%d blank frame(s) left unchanged" % nblank)
        # measure again: first, middle and last frame
        chk = sorted(set([0, n // 2, n - 1]))
        after = [relative(profile(out[i][..., :3]), pa) for i in chk if not _is_blank(out[i][..., :3])]
        before = [rels_before[k] for k in (keys[0], keys[len(keys) // 2], keys[-1])]
        avg = lambda rs, key: float(np.mean([r[key] for r in rs])) if rs else 0.0
        sb = float(np.mean([quality_score(r) for r in before]))
        sa = float(np.mean([quality_score(r) for r in after])) if after else 0.0
        worst_after = min(after, key=quality_score) if after else before[-1]
        verdict = _verdict(sa, worst_after)
        drop = before[0]["texture"] - before[-1]["texture"]
        if drop > 0.12:
            notes.append("texture fell %d%% from the first to the last frame before correction" % round(100 * drop))
        fl = quality_flags(worst_after)
        rep = ("%s: %s | quality %d -> %d of 100 | before: %s | after: %s%s%s | %.1f s" % (
            label, verdict, round(sb), round(sa), _fmt({k: avg(before, k) for k in before[0]}),
            _fmt({k: avg(after, k) for k in before[0]}) if after else "-",
            (" | still: " + "; ".join(fl)) if fl else "", (" | " + "; ".join(notes)) if notes else "",
            time.time() - t0))
        if verdict == "RE-RENDER":
            rep += " | re-render this scene with a new take, or raise strength"
        _log({"node": "SceneQualityLock", "label": label, "frames": n, "score_before": round(sb, 1),
              "score_after": round(sa, 1), "texture_before": round(avg(before, "texture"), 3),
              "texture_after": round(avg(after, "texture"), 3), "saturation_before": round(avg(before, "saturation"), 3),
              "saturation_after": round(avg(after, "saturation"), 3), "grain_before": round(avg(before, "grain"), 3),
              "grain_after": round(avg(after, "grain"), 3), "contrast_before": round(avg(before, "contrast"), 3),
              "contrast_after": round(avg(after, "contrast"), 3), "verdict": verdict,
              "notes": "; ".join(fl + notes)}, rep, write_log)
        return (_to_tensor(out), rep, float(sa))


class QualityLock:
    """One frame: correct it toward the gold frame (texture, grain, saturation, contrast)."""

    @classmethod
    def INPUT_TYPES(cls):
        req = {"image": ("IMAGE",), "anchor": ("IMAGE", {"tooltip": "The gold frame."}),
               "label": ("STRING", {"default": "Hand-off"})}
        req.update(dict(_SETTINGS_INPUTS))
        return {"required": req}

    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT")
    RETURN_NAMES = ("image", "report", "quality")
    FUNCTION = "run"
    CATEGORY = "video/ripcurlsurf"

    def run(self, image, anchor, label, strength, texture, max_texture_gain, saturation_limit, saturation, contrast,
            grain, grain_seed, write_log):
        im = _Checked(image, "image")
        gold = _Checked(anchor, "anchor")
        s = _settings(strength, texture, max_texture_gain, saturation_limit, saturation, contrast, grain)
        pa, notes = _gold(gold, im.h, im.w)
        outs, sb, sa, last_rel = [], [], [], None
        for i in range(im.n):
            f = im.frame(i)
            p = profile(f)
            rb = relative(p, pa)
            g = f if p["blank"] else apply_correction(f, plan_correction(p, pa, s), seed=int(grain_seed) + i)
            ra = relative(profile(g), pa)
            sb.append(quality_score(rb))
            sa.append(quality_score(ra))
            last_rel = (rb, ra)
            outs.append(np.concatenate([g, im.alpha(i)], axis=-1) if im.channels == 4 else g)
        notes = gold.summary() + im.summary() + notes
        rb, ra = last_rel
        verdict = _verdict(min(sa), ra)
        rep = "%s: %s | quality %d -> %d | before: %s | after: %s%s" % (
            label, verdict, round(np.mean(sb)), round(np.mean(sa)), _fmt(rb), _fmt(ra),
            (" | " + "; ".join(quality_flags(ra) + notes)) if (quality_flags(ra) or notes) else "")
        _log({"node": "QualityLock", "label": label, "frames": im.n, "score_before": round(float(np.mean(sb)), 1),
              "score_after": round(float(np.mean(sa)), 1), "texture_before": round(rb["texture"], 3),
              "texture_after": round(ra["texture"], 3), "saturation_before": round(rb["saturation"], 3),
              "saturation_after": round(ra["saturation"], 3), "grain_before": round(rb["grain"], 3),
              "grain_after": round(ra["grain"], 3), "contrast_before": round(rb["contrast"], 3),
              "contrast_after": round(ra["contrast"], 3), "verdict": verdict,
              "notes": "; ".join(quality_flags(ra) + notes)}, rep, write_log)
        return (_to_tensor(np.stack(outs, axis=0)), rep, float(np.mean(sa)))


class BestHandoffFrame:
    """Pick the frame the next scene starts from: the last frame, unless one of the few before it is clearly sharper."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "label": ("STRING", {"default": "Hand-off"}),
            "lookback": ("INT", {"default": 4, "min": 1, "max": 24, "step": 1,
                                 "tooltip": "How many of the last frames are compared (4 = the last 1/6 second at 24 fps)."}),
            "min_gain": ("FLOAT", {"default": 1.12, "min": 1.0, "max": 3.0, "step": 0.01,
                                   "tooltip": "An earlier frame is used only if it is this much sharper than the last one."}),
            "write_log": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("image", "index", "report")
    FUNCTION = "run"
    CATEGORY = "video/ripcurlsurf"

    def run(self, images, label, lookback, min_gain, write_log):
        clip = _Checked(images, "images")
        n = clip.n
        k = max(1, min(int(lookback), n))
        sharp = {}
        frames = {}
        for i in range(n - k, n):
            f = clip.frame(i)
            frames[i] = f
            if _is_blank(f):
                sharp[i] = -1.0
                continue
            y = _luma(f)
            d = _inner(y - _binom(y))[::2, ::2]
            sharp[i] = float(np.sqrt(np.mean(d * d)))
        last = n - 1
        good = [i for i in sharp if sharp[i] >= 0]
        notes = clip.summary()
        if not good:
            pick = last
            why = "every compared frame is blank"
        else:
            ref = max(good)                                   # the latest usable frame (continuity first)
            best = max(good, key=lambda i: (sharp[i], i))
            if best != ref and sharp[best] >= float(min_gain) * max(sharp[ref], 1e-9):
                pick = best
                why = "%d frame(s) before the end, %d%% sharper than frame %d" % (
                    last - pick, round(100 * (sharp[best] / max(sharp[ref], 1e-9) - 1)), ref + 1)
            else:
                pick = ref
                why = "the last frame" if ref == last else "the last frame is blank; the latest usable frame"
        rep = "%s: using frame %d of %d (%s)%s" % (label, pick + 1, n, why, (" | " + "; ".join(notes)) if notes else "")
        _log({"node": "BestHandoffFrame", "label": label, "frames": n, "verdict": "OK" if pick == last else "PICKED",
              "notes": rep}, rep, write_log)
        f = frames[pick]
        if clip.channels == 4:
            f = np.concatenate([f, clip.alpha(pick)], axis=-1)
        return (_to_tensor(f[None]), int(pick), rep)


def tg_s(res):
    return float(res[0].get("ts", 1.0)) if res else 1.0


_UNLINKED = object()           # default for lazy inputs: tells "not connected" apart from "connected, not made"


class QualityGate:
    """Choose the frame the next scene starts from.
    Candidate 1 is the quality-locked original: always a safe fallback, but its detail was only boosted, not rebuilt.
    Candidates 2-4 are image-model restores. Any candidate that moved or changed the picture is rejected; of the rest
    the best hand-off wins (soft counts five times as much as slightly crisp), and a restore is preferred over the
    boosted original unless the original is clearly better. Candidates 3 and 4 are lazy: ComfyUI makes them only when
    the best restore so far is still soft, so the stronger restores cost nothing on joins that are already good."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original": ("IMAGE", {"tooltip": "The hand-off frame before any restore: the layout every candidate must keep."}),
                "anchor": ("IMAGE", {"tooltip": "The gold frame."}),
                "candidate_1": ("IMAGE", {"tooltip": "The quality-locked original: always a safe fallback."}),
                "label": ("STRING", {"default": "Hand-off"}),
                "min_layout": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01,
                                         "tooltip": "Candidates whose large shapes moved more than this allows are rejected."}),
                "min_detail": ("FLOAT", {"default": 0.35, "min": -1.0, "max": 1.0, "step": 0.01,
                                         "tooltip": "Candidates whose faces, outlines or lettering changed more than this allows are rejected."}),
                "finish": ("BOOLEAN", {"default": True,
                                       "tooltip": "Give every candidate a light quality lock before comparing them (matches saturation and contrast)."}),
                "write_log": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "candidate_2": ("IMAGE", {"tooltip": "First image-model restore (always made)."}),
                "candidate_3": ("IMAGE", {"lazy": True,
                                          "tooltip": "Stronger restore. Made only when the best restore so far is still soft."}),
                "candidate_4": ("IMAGE", {"lazy": True,
                                          "tooltip": "Strongest restore. Made only when candidate 3 is still soft and was an improvement."}),
                "opening": ("IMAGE", {"tooltip": "The first frame of the clip this hand-off ends (the same shot). It sets how much "
                                                 "detail this kind of shot should have, so close-ups and soft backgrounds are not "
                                                 "pushed to the detail of a wide gold frame."}),
                "prefer_restored": ("BOOLEAN", {"default": True,
                                                "tooltip": "Kept for older workflows; it no longer changes the choice. Since 17 Oct 2026 the "
                                                           "real video frame (candidate 1) is kept unless it is clearly soft for this shot "
                                                           "AND a restore is clearly better: every image-model pass pulls the film toward "
                                                           "the image model's clean, illustrated look."}),
                "escalate_below": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.2, "step": 0.01,
                                             "tooltip": "Candidates 3 and 4 are made only while the best restore has less than this "
                                                        "share of the target detail."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("image", "report", "chosen", "quality")
    FUNCTION = "run"
    CATEGORY = "video/ripcurlsurf"

    # ---------------------------------------------------------------- shared by check_lazy_status and run
    @staticmethod
    def _prepare(original, anchor, opening):
        org = _Checked(original, "original")
        gold = _Checked(anchor, "anchor")
        o = org.frame(0)
        h, w = org.h, org.w
        pa, notes = _gold(gold, h, w)
        # Targets for this shot. Without an opening frame: the gold frame itself. With one: the opening frame's own
        # detail and edges, kept within 0.9-1.0 (detail) and 0.9-1.05 (edges) of the gold frame, so a close-up or a
        # soft background is not pushed to the detail of a wide gold frame, yet the chain can never slide.
        # The ceiling for detail is set from how much detail this clip lost from its first frame to its last: the
        # next clip will lose about as much, so it may start a little above the target (at most 15%) and its middle
        # lands on the target. Without an opening frame: 12%.
        tg = {"t": 1.0, "e": 1.0, "s": 1.0, "s_open": None, "t_open": None, "e_open": None, "hi": _HANDOFF_HIGH,
              "loss": None}
        if opening is not None:
            try:
                oc = _Checked(opening, "opening")
                f = oc.frame(0)
                if (oc.h, oc.w) != (h, w):
                    f = np.clip(_resize(f, h, w), 0, 1)
                po = profile(f)
                if po["blank"]:
                    notes.append("the opening frame is blank and was not used")
                else:
                    ro = relative(po, pa)
                    # surface target: the opening frame's own surface texture, but never below 90% of the gold frame
                    tg.update({"s_open": ro["surface"], "s": float(min(1.0, max(0.9, ro["surface"])))})
                    tg.update({"t_open": ro["texture"], "e_open": ro["edges"],
                               "t": float(min(1.0, max(0.9, ro["texture"]))), "e": float(min(1.05, max(0.9, ro["edges"])))})
                    keep = float(min(1.0, max(0.6, relative(profile(o), pa)["texture"] / max(ro["texture"], 1e-3))))
                    # the footage lock wins back about half of the loss, so half of it (in log terms) is pre-compensated
                    tg.update({"loss": 1.0 - keep, "hi": float(min(1.15, max(1.0, keep ** -0.25)))})
            except Exception as e:
                notes.append("opening frame skipped: %s" % e)
        return org, gold, o, h, w, pa, notes, tg

    @staticmethod
    def _score(o, pa, h, w, cands, finish, min_layout, min_detail, tg, notes):
        # Hand-off frames stay grain-free: grain fed into the next generation builds up into blotches. Restores that
        # came out crisper than the film are toned down to just above the target, and edges harder than the shot's
        # own are softened, so no clip starts over-sharpened and hard edges cannot pile up from join to join.
        hi_t, hi_e = tg["hi"] * tg["t"], 1.05 * tg["e"]
        # saturation: the video model adds 10-35% per clip (test 17 Oct 2026), so the start frame of the next clip is
        # brought back to within 6% of the gold frame, not 15%, or the colour climbs from clip to clip
        light = _settings(strength=0.6, texture=0.6, saturation=1.0, saturation_limit=1.06, contrast=0.3, grain=0.6,
                          texture_high=hi_t, high_pull=1.0, edges_high=hi_e, edges_pull=0.8)
        # one band filter applied to a picture does not move the measured band by the full amount, so the
        # toning-down is repeated (reduce only) until detail and edges are really back under their ceilings
        down = _settings(strength=1.0, texture=0.0, saturation=0.0, contrast=0.0, grain=0.0,
                         texture_high=hi_t, high_pull=1.0, edges_high=hi_e, edges_pull=1.0)
        res = []
        for k, c in cands:
            try:
                cc = _Checked(c, "candidate_%d" % k)
                f = cc.frame(0)
                if (cc.h, cc.w) != (h, w):
                    f = np.clip(_resize(f, h, w), 0, 1)
                    notes.append("candidate %d was %dx%d and was resized" % (k, cc.w, cc.h))
                notes += cc.summary()
            except Exception as e:
                notes.append("candidate %d skipped: %s" % (k, e))
                continue
            if _is_blank(f):
                notes.append("candidate %d is blank" % k)
                continue
            lay = layout_similarity(o, f)
            det = detail_similarity(o, f)
            p = profile(f)
            raw = relative(p, pa)
            if finish:
                g = apply_correction(f, plan_correction(p, pa, light), seed=99 + k)
                pg = profile(g)
                rel = relative(pg, pa)
                for _ in range(3):
                    if rel["texture"] <= 1.02 * hi_t and rel["edges"] <= 1.02 * hi_e:
                        break
                    g = apply_correction(g, plan_correction(pg, pa, down), seed=99 + k)
                    pg = profile(g)
                    rel = relative(pg, pa)
            else:
                g, rel = f, raw
            d = handoff_distance(rel, tg["t"], tg["e"], tg["s"])
            res.append({"k": k, "lay": lay, "det": det, "d": d, "score": 100.0 * math.exp(-d), "img": g, "rel": rel,
                        "raw": raw, "ts": tg["s"], "tt": tg["t"], "ok": lay >= float(min_layout) and det >= float(min_detail)})
        return res

    @staticmethod
    def _choose(res, prefer_restored):
        passing = [r for r in res if r["ok"]]
        if not passing:
            return max(res, key=lambda r: (r["lay"] + r["det"], -r["k"])), "CHECK", \
                "no candidate kept the picture within the limits; kept the most faithful one"
        best = min(passing, key=lambda r: (round(r["d"], 3), r["k"]))
        orig = next((r for r in passing if r["k"] == 1), None)
        restored = [r for r in passing if r["k"] > 1]
        # Every image-model pass pulls the start frame toward the image model's own clean, illustrated look, and the
        # video model builds on it: one pass per join added up to a painted film (test 17 Oct 2026: a restore was used
        # at 8 of 8 joins). The real video frame is kept unless it is clearly soft for this shot AND a restore is
        # clearly better.
        if orig is not None:
            soft = orig["raw"]["texture"] < 0.85 * float(orig.get("tt", 1.0))
            if best["k"] != 1:
                if soft and best["d"] + _FALLBACK_MARGIN < orig["d"]:
                    return best, "OK", "restore #%d used: the video frame was soft (detail %.2f)" % (best["k"], orig["raw"]["texture"])
                return orig, "OK", "the real video frame was kept (restore #%d was not clearly better)" % best["k"]
            return orig, "OK", "the real video frame was kept"
        return best, "OK", ""

    @staticmethod
    def _more(res, tg, escalate_below):
        """Is a stronger restore worth making? Returns (yes/no, reason)."""
        restored = sorted([r for r in res if r["k"] > 1], key=lambda r: r["k"])
        if not restored:
            return True, "no restore made yet"
        okr = [r for r in restored if r["ok"]]
        if not okr:
            return False, "the restores already changed the picture; a stronger one would change it more"
        if not restored[-1]["ok"]:
            return False, "the stronger restore changed the picture"
        best = min(okr, key=lambda r: r["d"])
        orig = next((r for r in res if r["k"] == 1), None)
        if orig is not None and orig["raw"]["texture"] >= 0.85 * tg["t"]:
            return False, "the video frame is sharp enough (detail %.2f); no stronger restore needed" % orig["raw"]["texture"]
        goal = float(escalate_below) * tg["t"]
        if best["raw"]["texture"] >= goal:
            return False, "the best restore is sharp enough (detail %.2f, goal %.2f)" % (best["raw"]["texture"], goal)
        if len(restored) >= 2 and restored[-1]["raw"]["texture"] < 1.03 * restored[-2]["raw"]["texture"]:
            return False, "the stronger restore added no detail"
        return True, "the best restore is still soft (detail %.2f, goal %.2f)" % (best["raw"]["texture"], goal)

    # ---------------------------------------------------------------- ComfyUI lazy evaluation
    def check_lazy_status(self, **kw):
        """Called by ComfyUI before run(): returns the lazy candidates still needed (one at a time)."""
        try:
            pending = [n for n in ("candidate_3", "candidate_4") if n in kw and kw[n] is None]
            if not pending or any(kw.get(n) is None for n in ("original", "anchor", "candidate_1")):
                return []
            org, gold, o, h, w, pa, notes, tg = self._prepare(kw["original"], kw["anchor"], kw.get("opening"))
            cands = [(k, kw.get("candidate_%d" % k)) for k in (1, 2, 3, 4)]
            res = self._score(o, pa, h, w, [(k, c) for k, c in cands if c is not None], bool(kw.get("finish", True)),
                              float(kw.get("min_layout", 0.75)), float(kw.get("min_detail", 0.35)), tg, [])
            need, why = self._more(res, tg, float(kw.get("escalate_below", 0.95)))
            if need:
                print("[QualityLock] %s: making %s (%s)" % (kw.get("label", "Quality Gate"),
                                                             pending[0].replace("_", " "), why))
                return pending[:1]
            return []
        except Exception as e:
            print("[QualityLock] Quality Gate: escalation check skipped (%s)" % e)
            return []

    def run(self, original, anchor, candidate_1, label, min_layout, min_detail, finish, write_log,
            candidate_2=None, candidate_3=_UNLINKED, candidate_4=_UNLINKED, opening=None, prefer_restored=True,
            escalate_below=0.95):
        skipped = [k for k, c in ((3, candidate_3), (4, candidate_4)) if c is None]     # connected, not needed
        c3 = None if candidate_3 is _UNLINKED else candidate_3
        c4 = None if candidate_4 is _UNLINKED else candidate_4
        org, gold, o, h, w, pa, notes, tg = self._prepare(original, anchor, opening)
        res = self._score(o, pa, h, w, [(k, c) for k, c in ((1, candidate_1), (2, candidate_2), (3, c3), (4, c4))
                                        if c is not None], finish, min_layout, min_detail, tg, notes)
        rel0 = relative(profile(o), pa)
        if not res:
            rep = "%s: no usable candidate - passing the original through" % label
            _log({"node": "QualityGate", "label": label, "frames": 1, "verdict": "CHECK", "notes": rep}, rep, write_log)
            return (_to_tensor(o[None]), rep, 0, 0.0)
        best, verdict, why = self._choose(res, bool(prefer_restored))
        if verdict == "OK" and best["rel"]["texture"] < 0.8 * tg["t"]:
            verdict = "CHECK"
            notes.append("the hand-off is still soft: re-render this scene with a new take if the next scene looks soft")
        if tg["t_open"] is not None:
            notes.append("detail target %.2f, edge target %.2f of gold (from the opening frame of this shot: %.2f, %.2f); "
                         "the clip lost %d%% detail from start to end, so restores may start up to %d%% above the target" % (
                             tg["t"], tg["e"], tg["t_open"], tg["e_open"], round(100 * tg["loss"]), round(100 * (tg["hi"] - 1))))
        if skipped:
            need, reason = self._more(res, tg, escalate_below)
            notes.append(("stronger restore #%s not needed: %s" if not need else "stronger restore #%s not made: %s") % (
                ", #".join(str(k) for k in skipped), reason))
        made = [r["k"] for r in res]
        table = ", ".join("#%d %s (layout %.2f, detail %.2f, texture %.2f, hand-off quality %d)" % (
            r["k"], "ok" if r["ok"] else "REJECTED", r["lay"], r["det"], r["rel"]["texture"], round(r["score"]))
            for r in res)
        fl = quality_flags(best["rel"], use_grain=False)
        extra = ([why] if why else []) + fl + notes + gold.summary() + org.summary()
        rep = "%s: %s | using candidate %d | %s | %s%s" % (
            label, verdict, best["k"], table, _fmt(best["rel"]), (" | " + "; ".join(extra)) if extra else "")
        _log({"node": "QualityGate", "label": label, "frames": 1,
              "score_before": round(100.0 * math.exp(-handoff_distance(rel0, tg["t"], tg["e"], tg["s"])), 1),
              "score_after": round(best["score"], 1), "texture_before": round(rel0["texture"], 3),
              "texture_after": round(best["rel"]["texture"], 3), "saturation_before": round(rel0["saturation"], 3),
              "saturation_after": round(best["rel"]["saturation"], 3), "grain_after": round(best["rel"]["grain"], 3),
              "contrast_before": round(rel0["contrast"], 3), "contrast_after": round(best["rel"]["contrast"], 3),
              "verdict": verdict,
              "notes": "chose #%d of made %s%s; %s" % (best["k"], "/".join("#%d" % k for k in made),
                                                       ("; skipped " + "/".join("#%d" % k for k in skipped)) if skipped else "",
                                                       table)}, rep, write_log)
        return (_to_tensor(best["img"][None]), rep, int(best["k"]), float(best["score"]))


_ANCHOR_MODES = ["store", "load (fall back to the input)"]


class QualityAnchor:
    """The film's gold frame. 'store' saves it to ComfyUI/output/quality/<name>.npy; 'load' reads it back
    (so a batch or a single-scene redo aims at the same target) and falls back to the input if it is missing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE", {"tooltip": "store: the gold frame. load: used only if nothing is stored yet."}),
            "name": ("STRING", {"default": "film"}),
            "mode": (_ANCHOR_MODES,),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")
    FUNCTION = "run"
    CATEGORY = "video/ripcurlsurf"

    @staticmethod
    def _paths(name):
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name or "")).strip("._") or "film"
        d = os.path.join(_out_dir(), "quality")
        return d, os.path.join(d, safe + ".npy"), os.path.join(d, safe + ".png"), safe

    @classmethod
    def IS_CHANGED(cls, image, name, mode):
        if str(mode).startswith("load"):
            p = cls._paths(name)[1]
            return os.path.getmtime(p) if os.path.isfile(p) else "missing"
        return "store"

    def run(self, image, name, mode):
        im = _Checked(image, "image")
        d, npy, png, safe = self._paths(name)
        f = im.frame(0)
        notes = im.summary()
        if str(mode).startswith("load"):
            if os.path.isfile(npy):
                try:
                    g = np.load(npy).astype(np.float32)
                    if g.ndim != 3 or g.shape[-1] != 3 or g.shape[0] < 16 or g.shape[1] < 16:
                        raise ValueError("unexpected shape %s" % (g.shape,))
                    g = np.clip(np.nan_to_num(g), 0, 1)
                    rep = "Quality Anchor: loaded the gold frame '%s' (%dx%d) from output/quality" % (safe, g.shape[1], g.shape[0])
                    _log({"node": "QualityAnchor", "label": safe, "frames": 1, "verdict": "LOADED", "notes": rep}, rep)
                    return (_to_tensor(g[None]), rep)
                except Exception as e:
                    notes.append("stored gold frame unreadable (%s)" % e)
            rep = ("Quality Anchor: no stored gold frame '%s' yet - using the input picture instead. Render scene 1 "
                   "once (it stores the gold frame) so every batch aims at the same quality." % safe)
            if notes:
                rep += " | " + "; ".join(notes)
            _log({"node": "QualityAnchor", "label": safe, "frames": 1, "verdict": "FALLBACK", "notes": rep}, rep)
            return (_to_tensor(f[None]), rep)
        p = profile(f)
        if p["blank"]:
            notes.append("WARNING: the gold frame is blank; quality locks will be weak")
        try:
            os.makedirs(d, exist_ok=True)
            np.save(npy, f)
            try:
                from PIL import Image
                Image.fromarray((f * 255.0 + 0.5).astype(np.uint8)).save(png)
            except Exception:
                pass
            rep = "Quality Anchor: stored the gold frame '%s' (%dx%d) in output/quality" % (safe, im.w, im.h)
        except Exception as e:
            rep = "Quality Anchor: could not store the gold frame (%s); it is still used for this run" % e
        if notes:
            rep += " | " + "; ".join(notes)
        _log({"node": "QualityAnchor", "label": safe, "frames": 1, "verdict": "STORED", "notes": rep}, rep)
        return (_to_tensor(f[None]), rep)


class QualityGateLegacy(QualityGate):
    """The same gate under its old id, so workflows made by earlier generator versions keep working.
    Hidden from the node menu; new workflows use QualityGate2."""
    DEPRECATED = True


# New workflows use the id QualityGate2: a ComfyUI that still has the old nodes then refuses the whole workflow
# at once ("node type not found") instead of failing at the first join after scene 1 has already been rendered.
NODE_CLASS_MAPPINGS = {
    "QualityAnchor": QualityAnchor,
    "SceneQualityLock": SceneQualityLock,
    "QualityLock": QualityLock,
    "BestHandoffFrame": BestHandoffFrame,
    "QualityGate2": QualityGate,
    "QualityGate": QualityGateLegacy,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "QualityAnchor": "Quality Anchor (gold frame for the whole film)",
    "SceneQualityLock": "Scene Quality Lock (whole clip, every frame)",
    "QualityLock": "Quality Lock (one frame)",
    "BestHandoffFrame": "Best Hand-off Frame (sharpest of the last frames)",
    "QualityGate2": "Quality Gate (the frame the next scene starts from)",
    "QualityGate": "Quality Gate (old id, for older workflows)",
}
