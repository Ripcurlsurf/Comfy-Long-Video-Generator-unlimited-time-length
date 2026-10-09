"""
sceneguard.py - ComfyUI custom node "Scene Guard" (package: ripcurlsurf_comfy).

Video models sometimes cut to a different shot, location or group of people in the middle
of a clip, even when the prompt describes one single scene. The prompt can only ask the
model not to do that. This node checks the finished frames and enforces it:

  * It looks for a sudden scene change inside the clip (a hard cut, or a dissolve to
    a different place) and throws away everything from that point on.
  * The kept part is always the shot that starts at the first frame, so it still joins
    cleanly to the first frame / previous scene. (Only if that first shot is very short
    does it keep the longest cut-free shot instead, and the report says so.)
  * Audio is trimmed to match, so sound never runs past the picture.
  * After trimming it either ends the clip early ("trim") or freezes the last good frame
    up to the original length ("hold_last_frame"), which keeps the planned timeline.
  * It writes one line per clip to  ComfyUI/output/scene_guard_log.txt  and returns the
    same text as "report", so you can see which scenes were cut and re-render only those
    with a new take number.

It cannot detect a character whose face or clothing slowly changes inside the same shot
(no model is used). It is plain numpy image statistics, no extra installs.
"""
import os
import time

import numpy as np
import torch

_GRID = 32


def _features(frames):
    """frames: float array [N,H,W,C] in 0..1 -> (small [N,g,g,3], hist [N,64])."""
    n, h, w, c = frames.shape
    if c < 3:
        frames = np.repeat(frames[..., :1], 3, axis=-1)
    g = max(2, min(_GRID, h, w))
    hh, ww = max(1, h // g), max(1, w // g)
    f = frames[:, : hh * g, : ww * g, :3]
    small = f.reshape(n, g, hh, g, ww, 3).mean(axis=(2, 4))
    q = np.clip((small * 4.0).astype(np.int32), 0, 3)
    idx = (q[..., 0] * 16 + q[..., 1] * 4 + q[..., 2]).reshape(n, -1)
    hist = np.zeros((n, 64), dtype=np.float32)
    for i in range(n):
        hist[i] = np.bincount(idx[i], minlength=64)[:64]
    hist /= float(idx.shape[1])
    return small.astype(np.float32), hist


def analyse(frames, fps, sensitivity=5, drift_limit=0.5, min_keep_seconds=1.0):
    """Return dict with keep range, events and a short description. frames: numpy [N,H,W,C]."""
    n = int(frames.shape[0])
    out = {"n": n, "start": 0, "end": n, "events": [], "note": "", "fallback": False}
    if n < 6:
        return out
    # features in chunks so very long clips do not need a second full copy
    smalls, hists = [], []
    for a in range(0, n, 32):
        s, h = _features(frames[a:a + 32])
        smalls.append(s)
        hists.append(h)
    small = np.concatenate(smalls, 0)
    hist = np.concatenate(hists, 0)

    f = 5.0 / float(min(10, max(1, sensitivity)))
    t_pix, t_hist = 0.10 * f, 0.22 * f
    dpix = np.abs(small[1:] - small[:-1]).mean(axis=(1, 2, 3))
    dhist = 0.5 * np.abs(hist[1:] - hist[:-1]).sum(axis=1)
    med = max(float(np.median(dpix)), 0.004)
    events = []
    for i in range(1, n):
        if dpix[i - 1] > t_pix and dhist[i - 1] > t_hist and dpix[i - 1] > 3.0 * med:
            events.append((i, "hard cut"))
    if drift_limit and drift_limit > 0:
        ref = hist[: min(4, n)].mean(axis=0)
        dist = 0.5 * np.abs(hist - ref).sum(axis=1)
        bad = np.nonzero(dist > float(drift_limit))[0]
        if bad.size and not (events and events[0][0] <= int(bad[0]) + 2):
            at = max(1, int(bad[0]) - 8)
            if not events or at < events[0][0]:
                events.append((at, "gradual change to a different scene"))
            events.sort()
    out["events"] = events
    if not events:
        return out
    min_keep = max(2, int(round(min_keep_seconds * fps)))
    first_end = events[0][0]
    if first_end >= min_keep:
        out["start"], out["end"] = 0, first_end
        return out
    cuts = [0] + [e[0] for e in events if e[1] == "hard cut"] + [n]
    segs = [(cuts[k], cuts[k + 1]) for k in range(len(cuts) - 1)]
    best = max(segs, key=lambda s: s[1] - s[0])
    out["start"], out["end"] = best
    out["fallback"] = best[0] != 0
    return out


def _slice_audio(audio, a_s, b_s, total_s, pad_to_total):
    wf = audio["waveform"]
    sr = int(audio["sample_rate"])
    ns = wf.shape[-1]
    a = max(0, min(ns, int(round(a_s * sr))))
    b = max(a, min(ns, int(round(b_s * sr))))
    cut = wf[..., a:b]
    if pad_to_total:
        want = int(round(total_s * sr))
        if cut.shape[-1] < want:
            pad = torch.zeros(cut.shape[:-1] + (want - cut.shape[-1],), dtype=cut.dtype, device=cut.device)
            cut = torch.cat([cut, pad], dim=-1)
    return {"waveform": cut, "sample_rate": sr}


class SceneGuard:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "label": ("STRING", {"default": "Scene"}),
                "sensitivity": ("INT", {"default": 5, "min": 1, "max": 10, "step": 1,
                                        "tooltip": "Higher catches smaller scene changes (and may flag fast action). 5 is a good start."}),
                "drift_limit": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                          "tooltip": "Also stop when the colours of the picture have moved this far from the opening frames (dissolves, slow changes of place). 0 = off (recommended; the default). Values below about 0.6 can cut clips with large camera moves."}),
                "min_keep_seconds": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 30.0, "step": 0.1}),
                "after_cut": (["trim", "hold_last_frame"],),
                "write_log": ("BOOLEAN", {"default": True}),
            },
            "optional": {"audio": ("AUDIO",)},
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING", "INT")
    RETURN_NAMES = ("images", "audio", "report", "frames_kept")
    FUNCTION = "guard"
    CATEGORY = "video/ripcurlsurf"

    def guard(self, images, fps, label, sensitivity, drift_limit, min_keep_seconds, after_cut, write_log, audio=None):
        n = int(images.shape[0])
        fps = float(fps) if fps else 24.0
        arr = images.detach().cpu().numpy() if hasattr(images, "detach") else images.cpu().numpy()
        r = analyse(arr, fps, sensitivity, drift_limit, min_keep_seconds)
        a, b = r["start"], r["end"]
        total_s = n / fps
        if not r["events"]:
            report = "%s: clean, no scene change in %d frames (%.2f s)" % (label, n, total_s)
            res = (images, audio, report, n)
        else:
            kept = images[a:b]
            if after_cut == "hold_last_frame" and b - a < n:
                tail = kept[-1:].repeat(n - (b - a), 1, 1, 1)
                kept = torch.cat([kept, tail], dim=0)
            ev = "; ".join("%s at frame %d (%.2f s)" % (k, i, i / fps) for i, k in r["events"][:4])
            report = "%s: SCENE CHANGE - %s. Kept frames %d-%d (%.2f s of %.2f s)%s%s. Re-render this scene with a new take number." % (
                label, ev, a, b, (b - a) / fps, total_s,
                ", ended early" if after_cut == "trim" else ", last good frame held to full length",
                "; the first shot was too short so the longest shot was kept instead" if r["fallback"] else "")
            aud = None
            if audio is not None:
                aud = _slice_audio(audio, a / fps, b / fps, total_s, after_cut == "hold_last_frame")
            res = (kept, aud, report, int(b - a))
        print("[SceneGuard] " + report)
        if write_log:
            try:
                import folder_paths
                p = os.path.join(folder_paths.get_output_directory(), "scene_guard_log.txt")
                with open(p, "a", encoding="utf-8") as fh:
                    fh.write(time.strftime("%Y-%m-%d %H:%M:%S") + " | " + report + "\n")
            except Exception:
                pass
        return res


NODE_CLASS_MAPPINGS = {"SceneGuard": SceneGuard}
NODE_DISPLAY_NAME_MAPPINGS = {"SceneGuard": "Scene Guard (cut off mid-clip scene changes)"}
