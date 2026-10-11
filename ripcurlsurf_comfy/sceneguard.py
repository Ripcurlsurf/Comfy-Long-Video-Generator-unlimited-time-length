"""
sceneguard.py - ComfyUI custom node "Scene Guard" (package: ripcurlsurf_comfy).

Video models sometimes cut to another shot, finish the action early and then freeze or
repeat it, or return a different number of frames than planned. A prompt can only ask the
model not to do this. This node checks the finished frames and enforces what it can:

  1. EXACT LENGTH   - with planned_frames set, the clip that leaves this node always has
                      exactly that many frames (longer is trimmed, shorter is filled by
                      holding the last good frame or by slowing the good part down), so
                      scene times never drift.
  2. SCENE CHANGES  - a sudden cut (and optionally a slow dissolve to another place) is
                      found and everything after it is removed.
  3. FROZEN / REPLAY- it also measures (and, if you choose, cuts off) a frozen ending and
                      a replayed action: the model doing the action once and then
                      repeating it.
  4. BEST OF N      - connect up to 3 more tries of the same scene (images_b, images_c,
                      images_d, each rendered with another seed). The try with the longest
                      clean stretch is used; the rest are dropped.
  5. REPORT         - one line per scene (also appended to ComfyUI/output/scene_guard_log.txt)
                      with the planned and real frame counts and every problem found.

It cannot see identity: a person whose face slowly changes inside one shot, or an action
repeated from a different camera angle, is not detected. It is plain numpy image
statistics (no model, no extra installs).
"""
import os
import time

import numpy as np
import torch

_GRID = 32
KINDS_CUT = ("hard cut", "gradual change to a different scene")


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


# ---- identity/location drift helpers ---------------------------------------
def _identity_signature(frames):
    """Small feature vector of the character region: RGB skin stats + frame mean."""
    n, h, w, c = frames.shape
    if c < 3:
        return None
    ch, cw = h // 2, w // 2
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    crop = frames[:, y0:y0 + ch, x0:x0 + cw, :3]
    r, g, b = crop[..., 0], crop[..., 1], crop[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    skin = (r > 0.30) & (r > g) & (g > b) & ((r - b) > 0.08) & (mx > 0.20) & (mx < 0.98)
    out = np.zeros((n, 8), dtype=np.float32)
    for i in range(n):
        sk = crop[i][skin[i]]
        if sk.size >= 32:
            out[i, 0:3] = sk.mean(axis=0)
            out[i, 3:6] = sk.std(axis=0)
            out[i, 6] = float(skin[i].mean())
        out[i, 7] = float(crop[i].mean())
    return out


def drift_to_reference(frames, ref_frames, fps):
    """How far the character signature drifts from the reference.
    Returns (score, per_frame). 0 = identical, higher = more drift."""
    sig = _identity_signature(frames)
    if sig is None:
        return 0.0, np.zeros(0, dtype=np.float32)
    ref = _identity_signature(ref_frames) if ref_frames is not None else None
    base = sig[: max(1, min(sig.shape[0], int(round(fps))))].mean(axis=0)
    refv = ref.mean(axis=0) if ref is not None and ref.size else base

    def _d(a, b):
        return (0.5 * np.abs(a[0:3] - b[0:3]).mean()
                + 0.5 * np.abs(a[3:6] - b[3:6]).mean()
                + 0.15 * abs(a[6] - b[6])
                + 0.05 * abs(a[7] - b[7]))

    per = np.array([_d(sig[i], refv) for i in range(sig.shape[0])], dtype=np.float32)
    k = max(1, int(round(fps * 0.5)))
    score = float(per[k:].max()) if per.shape[0] > k else float(per.max())
    return score, per


def _frozen_tail(small, fps):
    """Index where a frozen ending starts, or None.

    A frame counts as frozen when its change over 1 frame is tiny AND the change over 4
    frames is not bigger than the change over 1 frame (pure noise does not grow with the
    gap; real motion does)."""
    n = small.shape[0]
    if n < 12:
        return None
    m1 = np.zeros(n, dtype=np.float64)
    m4 = np.zeros(n, dtype=np.float64)
    m1[1:] = np.abs(small[1:] - small[:-1]).mean(axis=(1, 2, 3))
    m4[4:] = np.abs(small[4:] - small[:-4]).mean(axis=(1, 2, 3))
    m4[:4] = m4[4]
    m1[0] = m1[1]
    w = max(4, int(round(fps / 3.0)))
    ker = np.ones(w) / w
    W1 = np.convolve(m1, ker, mode="same")
    W4 = np.convolve(m4, ker, mode="same")
    froz = (W1 < 0.004) & (W4 < 1.5 * W1 + 0.0003)
    j = n
    while j > 0 and froz[j - 1]:
        j -= 1
    tail = n - j
    if tail >= max(int(fps), int(0.2 * n)) and j > int(fps):
        before = froz[:j]
        if float(np.mean(before)) < 0.5:
            return max(1, j - w // 2)
    return None


def _replay(small, fps):
    """Index where a replay of earlier motion starts, or None."""
    n = small.shape[0]
    stride = max(1, int(np.ceil(n / 600.0)))
    s = small[::stride]
    ns = s.shape[0]
    fs = fps / stride
    R = max(4, int(round(0.75 * fs)))
    L0 = max(R + 2, int(round(1.0 * fs)))
    d = max(2, int(round(0.4 * fs)))
    if ns < L0 + R + 2 * d + 2:
        return None
    g = s.shape[1]
    k = max(1, g // 8)
    v = s.reshape(ns, 8, k, 8, k, 3).mean(axis=(2, 4)).reshape(ns, -1) if g % 8 == 0 else s.reshape(ns, -1)
    D = np.empty((ns, ns), dtype=np.float32)
    for a in range(0, ns, 32):
        D[a:a + 32] = np.abs(v[a:a + 32, None, :] - v[None, :, :]).mean(axis=-1)
    mot = np.abs(v[1:] - v[:-1]).mean(axis=1)
    cm = np.concatenate([[0.0], np.cumsum(mot)])

    def win(lag):
        """mean distance D[i, i+lag] over windows of R frames, indexed by i."""
        if lag < 0 or lag >= ns:
            return None
        dg = np.diagonal(D, offset=lag).astype(np.float64)
        if dg.shape[0] < R:
            return None
        c = np.concatenate([[0.0], np.cumsum(dg)])
        return (c[R:] - c[:-R]) / R

    best = None
    for lag in range(L0, ns - R):
        e0, em, ep = win(lag), win(lag - d), win(lag + d)
        if e0 is None or em is None or ep is None:
            continue
        cnt = min(e0.shape[0], em.shape[0], ep.shape[0])
        if cnt <= 0:
            continue
        e0, em, ep = e0[:cnt], em[:cnt], ep[:cnt]
        moving = (cm[R:R + cnt] - cm[:cnt]) / R > 0.004
        ok = (e0 < 0.06) & (e0 < 0.5 * np.minimum(em, ep)) & moving
        idx = np.nonzero(ok)[0]
        if idx.size:
            j = int(idx[0]) + lag
            if best is None or j < best:
                best = j
    return None if best is None else int(best * stride)


def analyse(frames, fps, sensitivity=5, drift_limit=0.0, min_keep_seconds=1.0, extra=True):
    """frames: numpy [N,H,W,C]. Returns dict with events [(idx, kind)], good_end, cut_end."""
    n = int(frames.shape[0])
    out = {"n": n, "events": [], "good_end": n, "cut_end": n, "fallback": False, "start": 0, "end": n}
    if n < 6:
        return out
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
    if extra:
        fz = _frozen_tail(small, fps)
        if fz is not None:
            events.append((fz, "frozen ending"))
        rp = _replay(small, fps)
        if rp is not None:
            events.append((rp, "replayed action"))
    events.sort()
    out["events"] = events
    if events:
        out["good_end"] = events[0][0]
    cuts = [e for e in events if e[1] in KINDS_CUT]
    if cuts:
        out["cut_end"] = cuts[0][0]
    return out


def _segments(r, trim_extra):
    """Shots of one analysed clip as (start, end) frame ranges, split at every cut."""
    n = r["n"]
    use = [e for e in r["events"] if e[1] in KINDS_CUT]
    cuts = [0] + [e[0] for e in use] + [n]
    return [(cuts[k], cuts[k + 1]) for k in range(len(cuts) - 1) if cuts[k + 1] > cuts[k]]


def _choose_range(r, fps, min_keep_seconds, trim_extra):
    """Frame range [a,b) to keep for one analysed clip: the longest single shot,
    then (when asked) cut off a frozen ending or replay at the end of it."""
    n = r["n"]
    use = [e for e in r["events"] if e[1] in KINDS_CUT or trim_extra]
    if not use:
        return 0, n, False
    segs = _segments(r, trim_extra)
    a, b = max(segs, key=lambda s: (s[1] - s[0], -s[0]))
    if trim_extra:
        for i, k in r["events"]:
            if k not in KINDS_CUT and a < i < b:
                b = i
                break
    return a, b, a != 0


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
        elif cut.shape[-1] > want:
            cut = cut[..., :want]
    return {"waveform": cut, "sample_rate": sr}


def _slow_to_fit(kept, target):
    """Resample a clip to `target` frames by blending neighbouring frames (slow motion)."""
    arr = kept.detach().cpu().numpy() if hasattr(kept, "detach") else kept.cpu().numpy()
    k = arr.shape[0]
    if k <= 1 or target <= k:
        return kept
    pos = np.linspace(0.0, k - 1, target)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, k - 1)
    wgt = (pos - lo).astype(np.float32).reshape(-1, 1, 1, 1)
    out = arr[lo] * (1.0 - wgt) + arr[hi] * wgt
    return torch.from_numpy(out.astype(np.float32))


class SceneGuard:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "label": ("STRING", {"default": "Scene"}),
                "sensitivity": ("INT", {"default": 5, "min": 1, "max": 10, "step": 1,
                                        "tooltip": "Higher catches smaller scene changes. 5 is a good start."}),
                "drift_limit": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                          "tooltip": "Cut the clip when colours moved this far from its opening frames. 0 = off."}),
                "min_keep_seconds": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 30.0, "step": 0.1}),
                "after_cut": (["hold_last_frame", "slow_to_fit", "trim"],
                              {"tooltip": "What to do when a bad ending is removed."}),
                "trim_on": (["cuts only", "cuts, frozen endings and replays"],
                            {"tooltip": "Whether frozen endings and replays are also cut off (they are always measured)."}),
                "planned_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1,
                                           "tooltip": "Exact number of frames this scene must have. 0 = do not force."}),
                "write_log": ("BOOLEAN", {"default": True}),
                "drift_threshold": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "When a reference is supplied: report drift above this and prefer the least-drifted try. 0 = off."}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "images_b": ("IMAGE",), "audio_b": ("AUDIO",),
                "images_c": ("IMAGE",), "audio_c": ("AUDIO",),
                "images_d": ("IMAGE",), "audio_d": ("AUDIO",),
                "reference": ("IMAGE", {"tooltip": "Canonical character/location reference. Each try is scored for drift; the least-drifted try wins."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING", "INT")
    RETURN_NAMES = ("images", "audio", "report", "frames_out")
    FUNCTION = "guard"
    CATEGORY = "video/ripcurlsurf"

    def guard(self, images, fps, label, sensitivity, drift_limit, min_keep_seconds, after_cut, trim_on,
              planned_frames, write_log, drift_threshold=0.35, audio=None, images_b=None, audio_b=None,
              images_c=None, audio_c=None, images_d=None, audio_d=None, reference=None):
        fps = float(fps) if fps else 24.0
        trim_extra = trim_on != "cuts only"
        cands = [(images, audio)]
        for im, au in ((images_b, audio_b), (images_c, audio_c), (images_d, audio_d)):
            if im is not None:
                cands.append((im, au))
        ref_arr = reference.detach().cpu().numpy() if reference is not None else None
        res = []
        for im, au in cands:
            arr = im.detach().cpu().numpy() if hasattr(im, "detach") else im.cpu().numpy()
            r = analyse(arr, fps, sensitivity, drift_limit, min_keep_seconds, True)
            if ref_arr is not None and drift_threshold > 0:
                d, per = drift_to_reference(arr, ref_arr, fps)
                r["drift"] = d
                if d >= drift_threshold and per.size:
                    r["events"].append((int(np.argmax(per)), "identity/location drift (score %.2f)" % d))
                    r["events"].sort()
            res.append(r)

        def _key(k):
            seg = max(b - a for a, b in _segments(res[k], trim_extra))
            return (len(res[k]["events"]), -seg, res[k].get("drift", 0.0), k)
        order = sorted(range(len(cands)), key=_key)
        pick = order[0]
        im, au = cands[pick]
        r = res[pick]
        n = r["n"]
        a, b, fb = _choose_range(r, fps, min_keep_seconds, trim_extra)
        P = int(planned_frames) if planned_frames and planned_frames > 0 else 0
        target = P if P else n
        notes = []
        kept = im[a:b] if (a, b) != (0, n) else im
        kn = b - a
        if kn < target:
            if after_cut == "slow_to_fit" and kn > 1:
                want = min(target, int(kn * 1.5))
                kept = _slow_to_fit(kept, want)
                if want < target:
                    target = want
                    notes.append("good part slowed down 1.5x and the clip ends early; only %d usable frames of %d planned" % (kn, P))
                else:
                    notes.append("good part slowed down to fill the planned length")
            elif after_cut == "hold_last_frame":
                kept = torch.cat([kept, kept[-1:].repeat(target - kn, 1, 1, 1)], dim=0)
                notes.append("last good frame held to fill the planned length")
            else:
                notes.append("clip ends early")
        elif kn > target:
            kept = kept[:target]
            notes.append("trimmed to the planned length")
        out_n = int(kept.shape[0])
        # Audio always leaves with exactly out_n / fps seconds, in every after_cut mode,
        # so each scene's sound and pictures have the same length and nothing drifts
        # when scenes are joined.
        aud = None
        if au is not None:
            aud = _slice_audio(au, a / fps, b / fps, out_n / fps, True)
        else:
            other = next((x for _, x in cands if x is not None), None)
            if other is not None:
                # The winning try has no audio connected but other tries do: output
                # silence of the right length instead of nothing, so the film's audio
                # track does not lose a whole scene.
                sr = int(other["sample_rate"])
                shp = other["waveform"].shape[:-1] + (int(round(out_n / fps * sr)),)
                aud = {"waveform": torch.zeros(shp, dtype=other["waveform"].dtype), "sample_rate": sr}
                notes.append("the chosen try had no audio connected; silence was used")

        parts = []
        if len(cands) > 1:
            parts.append("tries: " + ", ".join(
                "#%d %s" % (k + 1, ("%d clean frames" % min(res[k]["good_end"], res[k]["n"]))
                            + ("" if not res[k]["events"] else " (" + res[k]["events"][0][1] + ")")
                            + ("" if "drift" not in res[k] else " drift %.2f" % res[k]["drift"]))
                for k in range(len(cands))) + " -> using #%d" % (pick + 1))
        if r["events"]:
            parts.append("problems: " + "; ".join("%s at frame %d (%.2f s)" % (k, i, i / fps) for i, k in r["events"][:4]))
        if "drift" in r:
            parts.append("drift score %.2f (threshold %.2f)" % (r["drift"], drift_threshold))
        if (a, b) != (0, n):
            parts.append("kept frames %d-%d" % (a, b))
        if fb:
            parts.append("the longest single shot was kept")
        parts += notes
        acted = any((k in KINDS_CUT or trim_extra) for _, k in r["events"])
        status = "OK" if not r["events"] else ("FIXED" if acted else "CHECK")
        report = "%s: %s | planned %s frames, got %d, output %d (%.2f s)%s" % (
            label, status, P if P else "-", n, out_n, out_n / fps, (" | " + " | ".join(parts)) if parts else "")
        if r["events"]:
            report += " | re-render this scene with a new take number or more tries"
        print("[SceneGuard] " + report)
        if write_log:
            try:
                import folder_paths
                p = os.path.join(folder_paths.get_output_directory(), "scene_guard_log.txt")
                with open(p, "a", encoding="utf-8") as fh:
                    fh.write(time.strftime("%Y-%m-%d %H:%M:%S") + " | " + report + "\n")
            except Exception:
                pass
        return (kept, aud, report, out_n)

NODE_CLASS_MAPPINGS = {"SceneGuard": SceneGuard}
NODE_DISPLAY_NAME_MAPPINGS = {"SceneGuard": "Scene Guard (exact length, cuts, best of N tries)"}
