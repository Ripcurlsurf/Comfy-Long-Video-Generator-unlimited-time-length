"""
refimagetools.py - ComfyUI custom nodes for reference pictures (package: ripcurlsurf_comfy).

Put the whole folder  ripcurlsurf_comfy  (this file plus __init__.py) in  ComfyUI/custom_nodes/  and restart ComfyUI.
It adds two nodes (category: image/reference):

A) Reference Photo Finish (remove painted look)   - see below
B) Load Image (optional, falls back to original)  - loads a picture from ComfyUI/input by
   file name; if the file is not there yet it passes the FALLBACK image through instead of
   stopping the whole workflow. Used for manual angle pictures: add the files whenever you
   have them, until then the original reference stands in.

A) Reference Photo Finish

What it does (pure image processing, no model, no extra dependencies):
  1. Colour match   - moves the colours and tone of the generated picture toward an
                      optional ORIGINAL reference image (done in Lab colour space), so
                      every angle of a character keeps the same skin, hair and clothing colours.
  2. Fine detail    - restores micro-contrast (skin pores, fabric, hair strands) that
                      diffusion editing smooths away.
  3. Film grain     - adds soft, midtone-weighted luminance grain. Real photographs have
                      sensor noise; flat noise-free surfaces are what reads as "painted".

It cannot change the pose, geometry or identity of the picture. It only removes the
smooth, over-saturated, noise-free look. Use low values first (defaults are mild).
"""
import numpy as np
import torch

# ---------------------------------------------------------------- helpers
_EPS = 1e-6


def _gauss_kernel(sigma):
    r = int(max(1, round(sigma * 3.0)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return (k / k.sum()).astype(np.float32)


def _blur(img, sigma):
    """Separable gaussian blur of an HxWxC float32 array (reflect padding)."""
    if sigma <= 0.05:
        return img
    k = _gauss_kernel(sigma)
    r = len(k) // 2
    h, w = img.shape[0], img.shape[1]
    r = min(r, h - 1, w - 1) if min(h, w) > 1 else 0
    if r < 1:
        return img
    k = _gauss_kernel(sigma)
    if len(k) // 2 != r:  # image smaller than the kernel: shrink the kernel
        x = np.arange(-r, r + 1, dtype=np.float32)
        k = np.exp(-(x * x) / (2.0 * sigma * sigma))
        k = (k / k.sum()).astype(np.float32)
    p = np.pad(img, ((r, r), (0, 0), (0, 0)), mode="reflect")
    a = np.zeros_like(img)
    for i, wt in enumerate(k):
        a += wt * p[i:i + h]
    p = np.pad(a, ((0, 0), (r, r), (0, 0)), mode="reflect")
    b = np.zeros_like(img)
    for i, wt in enumerate(k):
        b += wt * p[:, i:i + w]
    return b


def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1.0 / 2.4)) - 0.055)


_M = np.array([[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
_MI = np.linalg.inv(_M).astype(np.float32)
_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)


def _rgb_to_lab(rgb):
    lin = _srgb_to_linear(rgb).astype(np.float32)
    xyz = lin @ _M.T / _WHITE
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1).astype(np.float32)


def _lab_to_rgb(lab):
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    f = np.stack([fx, fy, fz], axis=-1)
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (f - 16.0 / 116.0) / 7.787) * _WHITE
    lin = xyz @ _MI.T
    return _linear_to_srgb(lin).astype(np.float32)


def _stats(ch):
    """Robust mean and std (ignores the darkest and brightest 2 percent)."""
    flat = ch.reshape(-1)
    lo, hi = np.percentile(flat, [2, 98])
    sel = flat[(flat >= lo) & (flat <= hi)]
    if sel.size < 16:
        sel = flat
    return float(sel.mean()), float(sel.std()) + _EPS


def _match_channel(ch, ref, amount, std_lo=0.6, std_hi=1.6):
    m1, s1 = _stats(ch)
    m2, s2 = _stats(ref)
    gain = float(np.clip(s2 / s1, std_lo, std_hi))
    out = (ch - m1) * gain + m2
    return ch * (1.0 - amount) + out * amount


def finish_one(img, ref=None, color_match=0.7, detail=0.6, grain=0.3,
               grain_size=1.0, seed=0):
    """img, ref: HxWx3 float32 in 0..1.  Returns HxWx3 float32 in 0..1."""
    img = np.clip(img.astype(np.float32), 0.0, 1.0)
    h, w = img.shape[:2]
    lab = _rgb_to_lab(img)

    # 1. colour and tone match to the original reference
    if ref is not None and color_match > 0:
        rlab = _rgb_to_lab(np.clip(ref.astype(np.float32), 0.0, 1.0))
        cm = float(np.clip(color_match, 0.0, 1.0))
        lab[..., 1] = _match_channel(lab[..., 1], rlab[..., 1], cm)
        lab[..., 2] = _match_channel(lab[..., 2], rlab[..., 2], cm)
        # tone is matched more gently: the angle may be lit differently
        lab[..., 0] = _match_channel(lab[..., 0], rlab[..., 0], cm * 0.5, 0.8, 1.25)

    # 2. fine detail (two scales of unsharp masking on luminance only)
    if detail > 0:
        L = lab[..., 0:1]
        scale = max(h, w) / 1024.0
        s1 = max(0.7, 1.1 * scale)
        s2 = max(1.8, 3.0 * scale)
        d1 = L - _blur(L, s1)
        d2 = _blur(L, s1) - _blur(L, s2)
        lab[..., 0:1] = L + detail * (0.9 * d1 + 0.5 * d2)

    # 3. film grain: soft luminance grain, strongest in the midtones
    if grain > 0:
        rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        n = rng.standard_normal((h, w, 1)).astype(np.float32)
        n = _blur(n, max(0.3, float(grain_size) * 0.6))
        n /= (n.std() + _EPS)
        Ln = np.clip(lab[..., 0:1] / 100.0, 0.0, 1.0)
        mid = 0.35 + 0.65 * (4.0 * Ln * (1.0 - Ln))
        lab[..., 0:1] += n * mid * (float(grain) * 3.2)
        c = rng.standard_normal((h, w, 2)).astype(np.float32)
        c = _blur(c, max(0.5, float(grain_size)))
        c /= (c.std() + _EPS)
        lab[..., 1:3] += c * (float(grain) * 0.5)

    lab[..., 0] = np.clip(lab[..., 0], 0.0, 100.0)
    return np.clip(_lab_to_rgb(lab), 0.0, 1.0)


# ---------------------------------------------------------------- node
class RefPhotoFinish:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "color_match": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                                          "tooltip": "How strongly the colours follow the original reference (needs the reference input)."}),
                "detail": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05,
                                     "tooltip": "Restores fine detail and micro-contrast. 0 = off."}),
                "grain": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                                    "tooltip": "Film grain amount. Noise-free flat surfaces read as painted."}),
                "grain_size": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 3.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
            },
            "optional": {
                "reference": ("IMAGE", {"tooltip": "The ORIGINAL reference picture. Colours are matched to it."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "image/reference"

    def run(self, image, color_match, detail, grain, grain_size, seed, reference=None):
        arr = image.detach().cpu().numpy().astype(np.float32)
        ref = None
        if reference is not None:
            ref = reference.detach().cpu().numpy().astype(np.float32)
        outs = []
        for i in range(arr.shape[0]):
            a = arr[i]
            rgb = a[..., :3]
            r = None
            if ref is not None:
                r = ref[min(i, ref.shape[0] - 1)][..., :3]
            res = finish_one(rgb, r, color_match, detail, grain, grain_size, int(seed) + i)
            if a.shape[-1] > 3:
                res = np.concatenate([res, a[..., 3:]], axis=-1)
            outs.append(res)
        return (torch.from_numpy(np.stack(outs, axis=0).astype(np.float32)),)


class RefLoadImageOptional:
    """Load an image from ComfyUI/input by name; use the fallback image if it is missing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fallback": ("IMAGE", {"tooltip": "Used when the file does not exist (the original reference)."}),
                "filename": ("STRING", {"default": "", "multiline": False,
                                        "tooltip": "File name inside ComfyUI/input, for example maya_back.png"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load"
    CATEGORY = "image/reference"

    @staticmethod
    def _path(filename):
        import os
        try:
            import folder_paths
            base = folder_paths.get_input_directory()
        except Exception:
            base = "input"
        name = str(filename or "").strip().replace("\\", "/")
        if not name or ".." in name.split("/"):
            return None
        p = os.path.join(base, name)
        return p if os.path.isfile(p) else None

    @classmethod
    def IS_CHANGED(cls, fallback, filename):
        import os
        p = cls._path(filename)
        return os.path.getmtime(p) if p else "missing"

    def load(self, fallback, filename):
        p = self._path(filename)
        if p is None:
            return (fallback,)
        try:
            from PIL import Image, ImageOps
            im = ImageOps.exif_transpose(Image.open(p))
            if im.mode in ("I;16", "I", "F"):
                im = im.point(lambda v: v * (1.0 / 256.0)).convert("L")
            im = im.convert("RGB")
            arr = np.asarray(im).astype(np.float32) / 255.0
            return (torch.from_numpy(arr[None, ...]),)
        except Exception:
            return (fallback,)


# ---------------------------------------------------------------- C) scene anchor
class SceneAnchor:
    """Re-anchor a chained frame to the canonical reference image(s).

    A diffusion model that starts each scene from the previous last frame slowly
    drifts: face, wardrobe, location and lighting change scene after scene. This
    node fights that by pulling the incoming chained frame back toward one to
    three canonical reference images. It never changes pose or motion:

      1. colour/tone match to the primary reference (Lab stats match)
      2. extra pull on skin-tone regions (face/hands) so identity snaps back
      3. gentle low-frequency structure blend, so silhouettes and large shapes
         pull toward the reference without freezing motion

    Use it on every soft-change chained frame in a long film. It pairs with
    Re-anchor every N scenes in the generator: the anchor node fixes drift
    between re-anchors; the re-anchor clears the residual drift completely.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "color_strength": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How hard colours and tone are pulled back to the reference. 0.5 to 0.7 is a good range for long films."}),
                "identity_strength": ("FLOAT", {"default": 0.40, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Extra pull on skin-tone regions (face/hands). Higher snaps identity back harder, but very high can ghost the reference's pose over the frame."}),
                "structure_strength": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.6, "step": 0.05,
                    "tooltip": "Low-frequency blend of large shapes and silhouettes toward the reference. Keep low; this only stops slow geometric drift."}),
                "detail": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Restores fine detail and micro-contrast after the pull-back. 0 = off."}),
                "grain": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05}),
                "grain_size": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 3.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
            },
            "optional": {
                "reference": ("IMAGE", {"tooltip": "Primary character or location reference (colour and structure anchor)."}),
                "reference_b": ("IMAGE", {"tooltip": "Second reference (for example the other character in the shot)."}),
                "reference_c": ("IMAGE", {"tooltip": "Third reference."}),
                "mask": ("MASK", {"tooltip": "Optional: 1 where the character is, 0 for background. If empty, a soft skin-tone mask is estimated from the reference."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "image/reference"

    @staticmethod
    def _skin_mask(rgb):
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        skin = (r > 0.30) & (r > g) & (g > b) & ((r - b) > 0.08) & (mx > 0.20) & (mx < 0.98) & ((mx - mn) > 0.05)
        return skin.astype(np.float32)

    @staticmethod
    def _resize_to(rgb, h, w):
        ih, iw = rgb.shape[:2]
        if (ih, iw) == (h, w):
            return rgb
        yi = (np.arange(h) * (ih / h)).astype(np.int32)
        xi = (np.arange(w) * (iw / w)).astype(np.int32)
        return rgb[yi][:, xi]

    @staticmethod
    def _match_lab(a, ref, amount, l_amount=None):
        la = _rgb_to_lab(a)
        lr = _rgb_to_lab(ref)
        la[..., 1] = _match_channel(la[..., 1], lr[..., 1], amount)
        la[..., 2] = _match_channel(la[..., 2], lr[..., 2], amount)
        la[..., 0] = _match_channel(la[..., 0], lr[..., 0],
                                    amount if l_amount is None else l_amount, 0.8, 1.25)
        la[..., 0] = np.clip(la[..., 0], 0.0, 100.0)
        return np.clip(_lab_to_rgb(la), 0.0, 1.0)

    def run(self, image, color_strength, identity_strength, structure_strength, detail, grain,
            grain_size, seed, reference=None, reference_b=None, reference_c=None, mask=None):
        arr = image.detach().cpu().numpy().astype(np.float32)
        refs = []
        for r in (reference, reference_b, reference_c):
            if r is not None:
                refs.append(r.detach().cpu().numpy().astype(np.float32))
        user_mask = mask.detach().cpu().numpy().astype(np.float32) if mask is not None else None

        outs = []
        for i in range(arr.shape[0]):
            a = arr[i, ..., :3]
            h, w = a.shape[:2]
            out = a.copy()

            if refs:
                ref0 = self._resize_to(refs[0][min(i, refs[0].shape[0] - 1), ..., :3], h, w)

                if color_strength > 0:
                    out = self._match_lab(out, ref0, float(color_strength),
                                          l_amount=float(color_strength) * 0.5)

                if identity_strength > 0:
                    # A global colour shift of the skin areas toward the reference's skin colour. The old version blended
                    # the reference's pixels in place; the reference is never aligned with the frame, so that averaged two
                    # different pictures (mush and ghosting) at every join - one of the causes of the waxy look.
                    if user_mask is not None:
                        m = self._resize_to(user_mask[..., None], h, w)[..., 0]
                    else:
                        m = self._skin_mask(out)
                        m = _blur(m[..., None], max(2.0, min(h, w) / 128.0))[..., 0]
                        m = np.clip(m * 2.0, 0.0, 1.0)
                    rm = self._skin_mask(ref0)
                    out_lab = _rgb_to_lab(out)
                    ref_lab = _rgb_to_lab(ref0)
                    if m.sum() > 64 and rm.sum() > 64:
                        sel = m > 0.5
                        rsel = rm > 0.5
                        d = [float(np.median(ref_lab[..., c][rsel]) - np.median(out_lab[..., c][sel])) for c in range(3)]
                        d[0] = float(np.clip(d[0], -8.0, 8.0)) * 0.5
                        d[1], d[2] = float(np.clip(d[1], -10.0, 10.0)), float(np.clip(d[2], -10.0, 10.0))
                        amt = float(identity_strength) * np.clip(m, 0.0, 1.0)
                        for c in range(3):
                            out_lab[..., c] = out_lab[..., c] + amt * d[c]
                        out_lab[..., 0] = np.clip(out_lab[..., 0], 0.0, 100.0)
                        out = np.clip(_lab_to_rgb(out_lab), 0.0, 1.0)

                if structure_strength > 0:
                    # only the broadest light and colour (very large blur), so it can never paint the reference in
                    s = float(structure_strength) * 0.5
                    big = max(16.0, min(h, w) / 8.0)
                    lo_out = _blur(out, big)
                    lo_ref = _blur(ref0, big)
                    hi_out = out - lo_out
                    lo_new = lo_out * (1 - s) + lo_ref * s
                    out = np.clip(lo_new + hi_out, 0.0, 1.0)

            out = finish_one(out, None, 0.0, float(detail), float(grain),
                             float(grain_size), int(seed) + i)
            if arr.shape[-1] > 3:
                out = np.concatenate([out, arr[i, ..., 3:]], axis=-1)
            outs.append(out)

        return (torch.from_numpy(np.stack(outs, axis=0).astype(np.float32)),)


# Node lists go last: every class above must already exist when Python reaches this.
NODE_CLASS_MAPPINGS = {
    "RefPhotoFinish": RefPhotoFinish,
    "RefLoadImageOptional": RefLoadImageOptional,
    "SceneAnchor": SceneAnchor,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RefPhotoFinish": "Reference Photo Finish (remove painted look)",
    "RefLoadImageOptional": "Load Image (optional, falls back to original)",
    "SceneAnchor": "Scene Anchor (identity-locked continuity)",
}
